from __future__ import annotations

import argparse
import csv
import json
import platform
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .checkpoint import load_model
from .decoder import OCDConfig, ordinal_constrained_decode
from .embeddings import load_embedding_pkl
from .labels import CLASS_NAMES, IGNORE_INDEX, build_cas_targets, load_cas_table, load_real_targets
from .metrics import add_transition_summary, confusion_matrix, count_transitions, metrics_from_confusion
from .model import OColT, OColTConfig
from .runtime import resolve_device


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "OColT_20260201.yaml"
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "OColT.pth"
DEFAULT_REAL_TEST = PACKAGE_ROOT / "splits" / "real_test.txt"
DEFAULT_CAS_TEST = PACKAGE_ROOT / "splits" / "cas_test.txt"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _read_ids(path: Path) -> list[str]:
    result: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.endswith((".pkl", ".csv")):
            value = value[:-4]
        result.append(value)
    if not result:
        raise ValueError(f"video list is empty: {path}")
    if len(result) != len(set(result)):
        raise ValueError(f"video list contains duplicate IDs: {path}")
    return result


@torch.inference_mode()
def _logits(
    model: OColT,
    embeddings: np.ndarray,
    stream_index: int,
    device: torch.device,
) -> torch.Tensor:
    if stream_index < 0 or stream_index >= embeddings.shape[0]:
        raise ValueError(
            f"stream_index={stream_index} is invalid for {embeddings.shape[0]} streams"
        )
    stream = torch.from_numpy(embeddings[stream_index]).unsqueeze(0).to(device)
    mask = torch.ones((1, stream.shape[1]), dtype=torch.float32, device=device)
    return model(stream.transpose(1, 2), mask).squeeze(0)


def _evaluate_one(
    *,
    dataset: str,
    model: OColT,
    device: torch.device,
    pkl_dir: Path,
    video_list: Path,
    label_path: Path,
    real_label_column: str,
    cas_fps: float,
    stream_index: int,
    ocd: OCDConfig,
) -> dict[str, Any]:
    if not pkl_dir.is_dir():
        raise FileNotFoundError(f"embedding directory not found: {pkl_dir}")
    if not video_list.is_file():
        raise FileNotFoundError(f"video list not found: {video_list}")
    if dataset == "real" and not label_path.is_dir():
        raise FileNotFoundError(f"REAL label directory not found: {label_path}")
    if dataset == "cas" and not label_path.is_file():
        raise FileNotFoundError(f"CAS label table not found: {label_path}")

    ids = _read_ids(video_list)
    cas_rows = load_cas_table(label_path) if dataset == "cas" else None
    aggregate = {
        "RAW": torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long),
        "OCD": torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long),
    }
    transitions: dict[str, list[int]] = {"RAW": [], "OCD": []}
    per_video: list[dict[str, Any]] = []
    total_frames = 0
    labeled_frames = 0

    for ordinal, video_id in enumerate(ids, start=1):
        pkl_path = pkl_dir / f"{video_id}.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(f"missing embedding: {pkl_path}")
        embeddings, image_names = load_embedding_pkl(pkl_path)
        length = embeddings.shape[1]
        if dataset == "real":
            target = load_real_targets(
                label_path / f"{video_id}.csv", image_names, real_label_column
            )
        else:
            assert cas_rows is not None
            if video_id not in cas_rows:
                raise KeyError(f"CAS VideoID {video_id!r} is absent from {label_path}")
            target = build_cas_targets(cas_rows[video_id], length, cas_fps)

        logits = _logits(model, embeddings, stream_index, device)
        predictions = {
            "RAW": logits.argmax(dim=1),
            "OCD": ordinal_constrained_decode(logits, ocd),
        }
        valid_count = int((target != IGNORE_INDEX).sum())
        if valid_count == 0:
            raise ValueError(f"video has no labeled frames: {video_id}")
        item: dict[str, Any] = {
            "video_id": video_id,
            "embedding_frames": int(length),
            "labeled_frames": valid_count,
            "ignored_frames": int(length - valid_count),
        }
        for mode, prediction in predictions.items():
            matrix = confusion_matrix(prediction, target)
            aggregate[mode] += matrix
            transition_count = count_transitions(prediction)
            transitions[mode].append(transition_count)
            item[mode] = {
                **metrics_from_confusion(matrix),
                "transitions": transition_count,
            }
        per_video.append(item)
        total_frames += int(length)
        labeled_frames += valid_count
        print(
            f"[{dataset.upper()} {ordinal:02d}/{len(ids):02d}] "
            f"{video_id}: frames={length} labeled={valid_count}",
            flush=True,
        )

    return {
        "videos_evaluated": len(ids),
        "embedding_frames": total_frames,
        "labeled_frames": labeled_frames,
        "ignored_frames": total_frames - labeled_frames,
        "metrics": {
            mode: add_transition_summary(
                metrics_from_confusion(aggregate[mode]), transitions[mode]
            )
            for mode in ("RAW", "OCD")
        },
        "per_video": per_video,
    }


def _summary_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, result in report["datasets"].items():
        for mode in ("RAW", "OCD"):
            metrics = result["metrics"][mode]
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "videos": result["videos_evaluated"],
                    "embedding_frames": result["embedding_frames"],
                    "labeled_frames": result["labeled_frames"],
                    "wF1": metrics["wF1"],
                    "wJacc": metrics["wJacc"],
                    "WMAPE": metrics["WMAPE"],
                    "transitions_mean": metrics["transitions_mean"],
                    "transitions_total": metrics["transitions_total"],
                    "stream_index": report["protocol"]["embedding_stream_index"],
                    "ocd_backward_penalty": report["protocol"]["ocd"]["backward_penalty"],
                    "ocd_skip_penalty": report["protocol"]["ocd"]["skip_penalty"],
                    "ocd_start_state": report["protocol"]["ocd"]["start_state"],
                    "ocd_smoothing_window": report["protocol"]["ocd"]["smoothing_window"],
                }
            )
    return rows


def _write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    csv_path = output_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    rows = _summary_rows(report)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


def _add_common_arguments(parser: argparse.ArgumentParser, output_dir: Path) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--stream-index", type=int, default=0)
    parser.add_argument("--ocd-backward", type=float)
    parser.add_argument("--ocd-skip", type=float)
    parser.add_argument("--ocd-start", type=int)
    parser.add_argument("--ocd-window", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OColT on one dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    datasets = parser.add_subparsers(dest="dataset", required=True)

    real = datasets.add_parser(
        "real",
        help="test on REAL-Colon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(real, Path("results/real"))
    real.add_argument("--pkl-dir", type=Path, required=True)
    real.add_argument("--video-list", type=Path, default=DEFAULT_REAL_TEST)
    real.add_argument("--gt-dir", type=Path, required=True)
    real.add_argument("--label-column", default="GT")

    cas = datasets.add_parser(
        "cas",
        help="test on CAS-Colon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(cas, Path("results/cas"))
    cas.add_argument("--pkl-dir", type=Path, required=True)
    cas.add_argument("--video-list", type=Path, default=DEFAULT_CAS_TEST)
    cas.add_argument("--label-csv", type=Path, required=True)
    cas.add_argument("--fps", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = _read_config(config_path)
    model_config = OColTConfig.from_mapping(config.get("model", {}))
    if model_config.output_size != len(CLASS_NAMES):
        raise ValueError("OColT common7 requires model.output_size=7")
    ocd_values = dict(config.get("ocd", {}))
    overrides = {
        "backward_penalty": args.ocd_backward,
        "skip_penalty": args.ocd_skip,
        "start_state": args.ocd_start,
        "smoothing_window": args.ocd_window,
    }
    ocd_values.update({key: value for key, value in overrides.items() if value is not None})
    ocd = OCDConfig.from_mapping(ocd_values)
    ocd.validate(model_config.output_size)

    device = resolve_device(args.device)
    model = load_model(checkpoint_path, OColT(model_config), device)
    if args.dataset == "real":
        result = _evaluate_one(
            dataset="real",
            model=model,
            device=device,
            pkl_dir=args.pkl_dir.expanduser(),
            video_list=args.video_list.expanduser(),
            label_path=args.gt_dir.expanduser(),
            real_label_column=args.label_column,
            cas_fps=5.0,
            stream_index=args.stream_index,
            ocd=ocd,
        )
    else:
        result = _evaluate_one(
            dataset="cas",
            model=model,
            device=device,
            pkl_dir=args.pkl_dir.expanduser(),
            video_list=args.video_list.expanduser(),
            label_path=args.label_csv.expanduser(),
            real_label_column="GT",
            cas_fps=args.fps,
            stream_index=args.stream_index,
            ocd=ocd,
        )
    datasets: dict[str, Any] = {args.dataset: result}

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": "OColT",
        "checkpoint": {
            "path": _display_path(checkpoint_path),
            "strict_state_dict_load": True,
        },
        "config": {"path": _display_path(config_path)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "device": str(device),
        "protocol": {
            "classes": list(CLASS_NAMES),
            "ignore_index": IGNORE_INDEX,
            "embedding_stream_index": args.stream_index,
            "pooled_frame_metrics": True,
            "ocd": {
                "backward_penalty": ocd.backward_penalty,
                "skip_penalty": ocd.skip_penalty,
                "start_state": ocd.start_state,
                "smoothing_window": ocd.smoothing_window,
            },
        },
        "datasets": datasets,
    }
    _write_report(report, args.output_dir.expanduser())
    for dataset, result in datasets.items():
        for mode in ("RAW", "OCD"):
            values = result["metrics"][mode]
            print(
                f"[{dataset.upper()}][{mode}] wF1={values['wF1']:.6f} "
                f"wJacc={values['wJacc']:.6f} WMAPE={values['WMAPE']:.6f} "
                f"transitions={values['transitions_mean']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
