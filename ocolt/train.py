from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .decoder import OCDConfig, ordinal_constrained_decode
from .embeddings import load_embedding_pkl
from .labels import CLASS_NAMES, IGNORE_INDEX, load_real_targets
from .metrics import confusion_matrix, metrics_from_confusion
from .model import OColT, OColTConfig
from .runtime import resolve_device


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "OColT_20260201.yaml"
DEFAULT_TRAIN_LIST = PACKAGE_ROOT / "splits" / "real_train.txt"
DEFAULT_VAL_LIST = PACKAGE_ROOT / "splits" / "real_val.txt"
N_CLASSES = len(CLASS_NAMES)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _read_ids(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.endswith((".pkl", ".csv")):
            value = value[:-4]
        values.append(value)
    if not values:
        raise ValueError(f"video list is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"video list contains duplicate IDs: {path}")
    return values


class RealDataset(Dataset):
    def __init__(
        self,
        pkl_dir: Path,
        gt_dir: Path,
        video_ids: list[str],
        label_column: str,
        random_stream: bool,
    ) -> None:
        self.pkl_dir = pkl_dir
        self.gt_dir = gt_dir
        self.video_ids = video_ids
        self.label_column = label_column
        self.random_stream = random_stream

    def __len__(self) -> int:
        return len(self.video_ids)

    def _load(self, index: int, stream_index: int | None) -> tuple[np.ndarray, np.ndarray, str]:
        video_id = self.video_ids[index]
        embeddings, image_names = load_embedding_pkl(self.pkl_dir / f"{video_id}.pkl")
        targets = load_real_targets(
            self.gt_dir / f"{video_id}.csv", image_names, self.label_column
        )
        keep = targets != IGNORE_INDEX
        if not np.any(keep):
            raise ValueError(f"video has no labeled frames: {video_id}")
        selected = (
            random.randrange(embeddings.shape[0])
            if stream_index is None
            else stream_index
        )
        if selected < 0 or selected >= embeddings.shape[0]:
            raise ValueError(
                f"stream {selected} is unavailable for {video_id}: "
                f"found {embeddings.shape[0]} streams"
            )
        return embeddings[selected, keep], targets[keep], video_id

    def load_targets(self, index: int) -> np.ndarray:
        return self._load(index, 0)[1]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        return self._load(index, None if self.random_stream else 0)


def _collate(
    batch: list[tuple[np.ndarray, np.ndarray, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(item[0].shape[0] for item in batch)
    dimension = batch[0][0].shape[1]
    embeddings = torch.zeros((len(batch), max_length, dimension), dtype=torch.float32)
    targets = torch.full((len(batch), max_length), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), max_length), dtype=torch.float32)
    names: list[str] = []
    for index, (features, labels, name) in enumerate(batch):
        length = features.shape[0]
        if features.shape[1] != dimension or len(labels) != length:
            raise ValueError(f"invalid embedding or target shape for {name}")
        embeddings[index, :length] = torch.from_numpy(features)
        targets[index, :length] = torch.from_numpy(labels)
        mask[index, :length] = 1.0
        names.append(name)
    return embeddings, targets, mask, names


def _class_weights(dataset: RealDataset) -> tuple[torch.Tensor, list[int]]:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for index in range(len(dataset)):
        target = dataset.load_targets(index)
        valid = target[(target >= 0) & (target < N_CLASSES)]
        counts += np.bincount(valid, minlength=N_CLASSES)[:N_CLASSES]
    if np.any(counts == 0):
        raise ValueError(f"training split contains an empty class: {counts.tolist()}")
    weights = np.sqrt(counts.sum() / counts.astype(np.float64))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5) & (targets != IGNORE_INDEX)
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(targets)
    return (values * valid).sum() / valid.sum().clamp_min(1)


def _tmse(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    valid = (mask > 0.5) & (targets != IGNORE_INDEX)
    valid_pairs = valid[:, 1:] & valid[:, :-1]
    log_probabilities = F.log_softmax(logits, dim=-1)
    differences = (log_probabilities[:, 1:] - log_probabilities[:, :-1]).pow(2)
    differences = differences.clamp(max=threshold**2)
    denominator = valid_pairs.sum().clamp_min(1) * logits.shape[-1]
    return (differences * valid_pairs.unsqueeze(-1)).sum() / denominator


def _dilate(values: torch.Tensor, radius: int) -> torch.Tensor:
    result = values.clone()
    for offset in range(1, radius + 1):
        result[:, offset:] |= values[:, :-offset]
        result[:, :-offset] |= values[:, offset:]
    return result


def _boundary_smoothness(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    valid = (mask > 0.5) & (targets != IGNORE_INDEX)
    changes = (targets[:, 1:] != targets[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundaries = torch.zeros_like(valid)
    boundaries[:, 1:] |= changes
    boundaries[:, :-1] |= changes
    boundaries = _dilate(boundaries, radius)
    stable = (
        valid[:, 1:]
        & valid[:, :-1]
        & ~boundaries[:, 1:]
        & ~boundaries[:, :-1]
    )
    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    forward = (
        probabilities[:, 1:]
        * (log_probabilities[:, 1:] - log_probabilities[:, :-1])
    ).sum(dim=-1)
    backward = (
        probabilities[:, :-1]
        * (log_probabilities[:, :-1] - log_probabilities[:, 1:])
    ).sum(dim=-1)
    divergence = 0.5 * (forward + backward)
    return (divergence * stable).sum() / stable.sum().clamp_min(1)


class OrdinalHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_CLASSES, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(logits).squeeze(-1)) * (N_CLASSES - 1)


def _ordinal_loss(
    head: OrdinalHead,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = (mask > 0.5) & (targets != IGNORE_INDEX)
    if not valid.any():
        return logits.sum() * 0.0
    prediction = head(logits)
    return F.l1_loss(prediction[valid], targets[valid].float())


@torch.inference_mode()
def _validate(
    model: OColT,
    loader: DataLoader,
    device: torch.device,
    ocd: OCDConfig,
) -> dict[str, Any]:
    model.eval()
    aggregate = {
        "RAW": torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.long),
        "OCD": torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.long),
    }
    videos = 0
    for embeddings, targets, mask, names in loader:
        embeddings = embeddings.to(device).transpose(1, 2)
        mask_device = mask.to(device)
        logits = model(embeddings, mask_device)
        for index, name in enumerate(names):
            length = int(mask[index].sum().item())
            target = targets[index, :length].numpy()
            if not np.any(target != IGNORE_INDEX):
                raise ValueError(f"validation video has no labeled frames: {name}")
            video_logits = logits[index, :length]
            predictions = {
                "RAW": video_logits.argmax(dim=-1),
                "OCD": ordinal_constrained_decode(video_logits, ocd),
            }
            for mode, prediction in predictions.items():
                aggregate[mode] += confusion_matrix(prediction, target)
            videos += 1
    return {
        "videos": videos,
        "RAW": metrics_from_confusion(aggregate["RAW"]),
        "OCD": metrics_from_confusion(aggregate["OCD"]),
    }


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    learning_rate: float,
    warmup_learning_rate: float,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    minimum = minimum_learning_rate / learning_rate
    warmup_start = warmup_learning_rate / learning_rate

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / warmup_steps
            return warmup_start + progress * (1.0 - warmup_start)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return minimum + 0.5 * (1.0 - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def _load_training_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"resume checkpoint must be a mapping: {path}")
    required = {
        "epoch",
        "validation",
        "selection_tuple",
        "model_state_dict",
        "ordinal_head_state_dict",
        "optimizer_state_dict",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"resume checkpoint is missing keys: {missing}")
    return value


def _save_checkpoint(
    path: Path,
    epoch: int,
    validation: Mapping[str, Any],
    selection: tuple[float, ...],
    model: OColT,
    ordinal_head: OrdinalHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "monitor": "REAL_validation_OCD_wJacc",
            "best_value": selection[0],
            "validation": validation,
            "selection_tuple": list(selection),
            "model_state_dict": dict(model.state_dict()),
            "ordinal_head_state_dict": dict(ordinal_head.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training OColT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pkl-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--train-video-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--val-video-list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--label-column", default="GT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _read_config(args.config.expanduser().resolve())
    model_config = OColTConfig.from_mapping(config.get("model", {}))
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("config.training must be a mapping")
    if model_config.output_size != N_CLASSES:
        raise ValueError(f"OColT requires {N_CLASSES} output classes")
    if training.get("class_weighting") != "sqrt_inverse":
        raise ValueError("training.class_weighting must be sqrt_inverse")
    if training.get("training_stream") != "random_single":
        raise ValueError("training.training_stream must be random_single")
    if int(training.get("validation_stream", 0)) != 0:
        raise ValueError("training.validation_stream must be 0")

    seed = int(training.get("seed", 20260845))
    deterministic = bool(training.get("deterministic", True))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(args.device)
    pkl_dir = args.pkl_dir.expanduser()
    gt_dir = args.gt_dir.expanduser()
    if not pkl_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError("--pkl-dir and --gt-dir must be directories")
    train_ids = _read_ids(args.train_video_list.expanduser())
    val_ids = _read_ids(args.val_video_list.expanduser())
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"train and validation splits overlap: {overlap}")

    train_dataset = RealDataset(
        pkl_dir, gt_dir, train_ids, args.label_column, random_stream=True
    )
    val_dataset = RealDataset(
        pkl_dir, gt_dir, val_ids, args.label_column, random_stream=False
    )
    class_weights, class_counts = _class_weights(train_dataset)
    class_weights = class_weights.to(device)

    batch_size = int(training.get("batch_size", 4))
    num_workers = int(training.get("num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    generator = torch.Generator().manual_seed(seed + 11)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(seed + 23),
    )

    model = OColT(model_config).to(device)
    ordinal_head = OrdinalHead(int(training.get("ordinal_hidden_size", 64))).to(device)
    parameters = [*model.parameters(), *ordinal_head.parameters()]
    learning_rate = float(training.get("learning_rate", 5e-4))
    minimum_learning_rate = float(training.get("minimum_learning_rate", 1e-6))
    warmup_learning_rate = float(training.get("warmup_learning_rate", 5e-7))
    scheduler_epochs = int(training.get("epochs", 100))
    epochs = int(args.epochs if args.epochs is not None else scheduler_epochs)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if scheduler_epochs < 1 or epochs > scheduler_epochs:
        raise ValueError("epochs cannot exceed training.epochs in the configuration")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=max(scheduler_epochs * len(train_loader), 1),
        warmup_steps=int(training.get("warmup_epochs", 5)) * len(train_loader),
        learning_rate=learning_rate,
        warmup_learning_rate=warmup_learning_rate,
        minimum_learning_rate=minimum_learning_rate,
    )

    resume: Mapping[str, Any] | None = None
    start_epoch = 0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        resume = _load_training_checkpoint(resume_path)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        ordinal_head.load_state_dict(resume["ordinal_head_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"])
        if start_epoch < 0:
            raise ValueError("resume checkpoint epoch must be non-negative")
        scheduler_state = resume.get("scheduler_state_dict")
        if isinstance(scheduler_state, Mapping):
            scheduler.load_state_dict(scheduler_state)
        else:
            completed_steps = start_epoch * len(train_loader)
            expected_lr = learning_rate * scheduler.lr_lambdas[0](completed_steps)
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isclose(actual_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "resume optimizer learning rate does not match the configured schedule"
                )
            scheduler.last_epoch = completed_steps
            scheduler._step_count = completed_steps + 1
            scheduler._last_lr = [float(group["lr"]) for group in optimizer.param_groups]
        print(f"[RESUME] {resume_path} epoch={start_epoch}", flush=True)

    if epochs <= start_epoch:
        raise ValueError(f"epochs must be greater than resume epoch {start_epoch}")

    ocd = OCDConfig.from_mapping(config.get("ocd", {}))
    ocd.validate(N_CLASSES)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "OColT.pth"
    history_path = output_dir / "validation_history.json"
    summary_path = output_dir / "training_summary.json"
    history: list[dict[str, Any]] = []
    best_selection: tuple[float, ...] | None = None
    best_validation: Mapping[str, Any] | None = None
    best_epoch = -1
    if resume is not None:
        selection_value = resume["selection_tuple"]
        validation_value = resume["validation"]
        if not isinstance(selection_value, (list, tuple)) or len(selection_value) != 5:
            raise ValueError("resume selection_tuple must contain five values")
        if not isinstance(validation_value, Mapping):
            raise ValueError("resume validation must be a mapping")
        best_selection = tuple(float(value) for value in selection_value)
        best_validation = validation_value
        best_epoch = start_epoch
        _save_checkpoint(
            checkpoint_path,
            best_epoch,
            best_validation,
            best_selection,
            model,
            ordinal_head,
            optimizer,
            scheduler,
        )

    print(f"[DATA] train={len(train_ids)} validation={len(val_ids)}")
    print(f"[DATA] class_counts={class_counts}")
    print(f"[DATA] class_weights={class_weights.detach().cpu().tolist()}")

    def validate(epoch: int) -> None:
        nonlocal best_selection, best_validation, best_epoch
        result = _validate(model, val_loader, device, ocd)
        candidate = (
            float(result["OCD"]["wJacc"]),
            float(result["OCD"]["wF1"]),
            float(result["RAW"]["wJacc"]),
            float(result["RAW"]["wF1"]),
            -float(epoch),
        )
        improved = best_selection is None or candidate > best_selection
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation": result,
                "improved": improved,
            }
        )
        _write_json(history_path, history)
        print(
            f"[VALID epoch={epoch:03d}] "
            f"RAW wF1={result['RAW']['wF1']:.6f} "
            f"wJacc={result['RAW']['wJacc']:.6f} | "
            f"OCD wF1={result['OCD']['wF1']:.6f} "
            f"wJacc={result['OCD']['wJacc']:.6f}",
            flush=True,
        )
        if improved:
            best_selection = candidate
            best_validation = result
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path,
                epoch,
                result,
                candidate,
                model,
                ordinal_head,
                optimizer,
                scheduler,
            )
            print(f"[BEST] epoch={epoch:03d}", flush=True)

    if resume is None:
        validate(0)
    ce_weight = float(training.get("cross_entropy_weight", 0.5))
    tmse_weight = float(training.get("tmse_weight", 0.15))
    tmse_threshold = float(training.get("tmse_threshold", 4.0))
    boundary_weight = float(training.get("boundary_smoothness_weight", 0.5))
    boundary_radius = int(training.get("boundary_radius", 5))
    ordinal_weight = float(training.get("ordinal_weight", 0.3))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    validation_every = int(training.get("validation_every", 1))
    if validation_every < 1:
        raise ValueError("training.validation_every must be positive")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        ordinal_head.train()
        totals = {"ce": 0.0, "tmse": 0.0, "boundary": 0.0, "ordinal": 0.0}
        for embeddings, targets, mask, _ in train_loader:
            embeddings = embeddings.to(device).transpose(1, 2)
            targets = targets.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings, mask)
            ce = _masked_cross_entropy(logits, targets, mask, class_weights)
            temporal = _tmse(logits, targets, mask, tmse_threshold)
            boundary = _boundary_smoothness(logits, targets, mask, boundary_radius)
            ordinal = _ordinal_loss(ordinal_head, logits, targets, mask)
            loss = (
                ce_weight * ce
                + tmse_weight * temporal
                + boundary_weight * boundary
                + ordinal_weight * ordinal
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            optimizer.step()
            scheduler.step()
            totals["ce"] += float(ce.detach().item())
            totals["tmse"] += float(temporal.detach().item())
            totals["boundary"] += float(boundary.detach().item())
            totals["ordinal"] += float(ordinal.detach().item())
        denominator = max(len(train_loader), 1)
        print(
            f"[TRAIN epoch={epoch:03d}] "
            f"ce={totals['ce']/denominator:.6f} "
            f"tmse={totals['tmse']/denominator:.6f} "
            f"boundary={totals['boundary']/denominator:.6f} "
            f"ordinal={totals['ordinal']/denominator:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.9g}",
            flush=True,
        )
        if epoch % validation_every == 0 or epoch == epochs:
            validate(epoch)

    _write_json(
        summary_path,
        {
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "checkpoint": checkpoint_path.name,
            "epochs": epochs,
        },
    )
    print(f"[DONE] {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
