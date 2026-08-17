from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from PIL import Image

from .embeddings import save_embedding_pkl
from .feature_encoder import (
    ENCODER_MODEL_NAME,
    IMAGE_SIZE,
    build_transform,
    encode_batch,
    load_encoder,
)
from .labels import parse_frame_number
from .runtime import resolve_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _read_ids(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.endswith((".pkl", ".csv", ".mp4")):
            value = value.rsplit(".", 1)[0]
        values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"video list must be non-empty and unique: {path}")
    return values


def _frame_directory(root: Path, video_id: str) -> Path:
    candidates = [root / f"{video_id}_frames", root / video_id]
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one frame directory for {video_id}: {candidates}"
        )
    return matches[0]


def _manifest_names(
    path: Path,
    column: str,
    source_fps: float,
    target_fps: float,
    sample_phase: int,
    label_column: str | None,
    keep_labels: set[str] | None,
) -> list[str]:
    ratio = source_fps / target_fps
    stride = int(round(ratio))
    if not math.isclose(ratio, stride, rel_tol=0, abs_tol=1e-8):
        raise ValueError("frame-manifest mode requires source_fps/target_fps to be an integer")
    names: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if column not in fields:
            raise ValueError(f"{path} lacks {column!r}; found {fields}")
        if label_column and label_column not in fields:
            raise ValueError(f"{path} lacks label column {label_column!r}")
        for row in reader:
            name = str(row.get(column, "")).strip()
            frame = parse_frame_number(name)
            if frame < 0:
                raise ValueError(f"cannot parse physical frame index from {name!r}")
            if (frame - sample_phase) % stride != 0:
                continue
            if label_column and keep_labels is not None:
                if str(row.get(label_column, "")).strip().lower() not in keep_labels:
                    continue
            names.append(name)
    if not names:
        raise ValueError(f"no sampled frames remained in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate frame_filename values in {path}")
    return names


def _encode_images(
    images: Iterable[tuple[str, Image.Image]],
    model,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    transform = build_transform()
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    features: list[np.ndarray] = []

    def flush() -> None:
        if tensors:
            batch = torch.stack(tensors)
            features.append(encode_batch(model, batch, device).numpy().astype(np.float32))
            tensors.clear()

    for name, image in images:
        names.append(name)
        tensors.append(transform(image.convert("RGB")))
        if len(tensors) >= batch_size:
            flush()
    flush()
    if not features:
        raise ValueError("no images were encoded")
    return np.concatenate(features, axis=0)[np.newaxis, ...], names


def _frame_images(frame_dir: Path, names: list[str]) -> Iterator[tuple[str, Image.Image]]:
    for name in names:
        path = frame_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing frame: {path}")
        with Image.open(path) as image:
            yield name, image.convert("RGB")


def _video_images(
    video_path: Path, video_id: str, target_fps: float
) -> tuple[Iterator[tuple[str, Image.Image]], dict]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required for video extraction") from error
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")
    if source_fps < target_fps:
        capture.release()
        raise ValueError(f"source fps {source_fps} is below target fps {target_fps}")
    wanted: list[int] = []
    sample = 0
    while True:
        frame = int(round(sample * source_fps / target_fps))
        if frame >= frame_count:
            break
        if not wanted or frame != wanted[-1]:
            wanted.append(frame)
        sample += 1

    def iterator() -> Iterator[tuple[str, Image.Image]]:
        wanted_pointer = 0
        current = 0
        try:
            while wanted_pointer < len(wanted):
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(
                        f"video ended before frame {wanted[wanted_pointer]}: {video_path}"
                    )
                if current == wanted[wanted_pointer]:
                    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                    name = f"{video_id}_{current:08d}.jpg"
                    yield name, Image.fromarray(rgb).convert("RGB")
                    wanted_pointer += 1
                current += 1
        finally:
            capture.release()

    return iterator(), {
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "target_fps": target_fps,
        "sampled_frames": len(wanted),
    }


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing per-video PKLs and embedding_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic stream-0 PKLs for OColT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    frames = commands.add_parser("frame-manifest")
    _common_parser(frames)
    frames.add_argument("--frames-root", type=Path, required=True)
    frames.add_argument("--manifest-dir", type=Path, required=True)
    frames.add_argument("--manifest-column", default="frame_filename")
    frames.add_argument("--source-fps", type=float, default=30.0)
    frames.add_argument("--sample-phase", type=int, default=0)
    frames.add_argument("--label-column")
    frames.add_argument(
        "--keep-labels",
        help="comma-separated labels retained when --label-column is supplied",
    )

    videos = commands.add_parser("video")
    _common_parser(videos)
    videos.add_argument("--video-dir", type=Path, required=True)
    videos.add_argument("--video-extension", default=".mp4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.target_fps <= 0:
        raise ValueError("batch-size and target-fps must be positive")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = resolve_device(args.device)
    model = load_encoder(args.encoder_checkpoint.expanduser(), device)
    video_ids = _read_ids(args.video_list.expanduser())
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = out_dir / "embedding_manifest.json"
    if manifest_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output manifest already exists: {manifest_output}; pass --overwrite to replace it"
        )
    records: list[dict] = []

    for ordinal, video_id in enumerate(video_ids, start=1):
        output = out_dir / f"{video_id}.pkl"
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"output PKL already exists: {output}; pass --overwrite to replace it"
            )
        if args.command == "frame-manifest":
            manifest = args.manifest_dir.expanduser() / f"{video_id}.csv"
            keep = (
                {value.strip().lower() for value in args.keep_labels.split(",") if value.strip()}
                if args.keep_labels
                else None
            )
            names = _manifest_names(
                manifest,
                args.manifest_column,
                args.source_fps,
                args.target_fps,
                args.sample_phase,
                args.label_column,
                keep,
            )
            frame_dir = _frame_directory(args.frames_root.expanduser(), video_id)
            images = _frame_images(frame_dir, names)
            sampling = {
                "mode": "frame-manifest",
                "source_fps": args.source_fps,
                "target_fps": args.target_fps,
                "sample_phase": args.sample_phase,
            }
        else:
            extension = args.video_extension
            if not extension.startswith("."):
                extension = "." + extension
            video_path = args.video_dir.expanduser() / f"{video_id}{extension}"
            images, sampling = _video_images(video_path, video_id, args.target_fps)
            sampling["mode"] = "video"

        embeddings, image_names = _encode_images(
            images, model, device, args.batch_size
        )
        metadata = {
            "schema_version": 1,
            "encoder": ENCODER_MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "feature_dimension": 384,
            "stream_index": 0,
            "deterministic": True,
            "sampling": sampling,
        }
        save_embedding_pkl(output, embeddings, image_names, metadata)
        records.append(
            {
                "video_id": video_id,
                "frames": len(image_names),
                "output": output.name,
                "sampling": sampling,
            }
        )
        print(
            f"[{ordinal:02d}/{len(video_ids):02d}] {video_id}: "
            f"shape={tuple(embeddings.shape)} -> {output}",
            flush=True,
        )

    embedding_manifest = {
        "schema_version": 1,
        "encoder": ENCODER_MODEL_NAME,
        "preprocess": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "resize_mode": "stretch",
            "interpolation": "bilinear",
            "normalization": "ImageNet mean/std",
            "pooling": "CLS token",
        },
        "device": str(device),
        "videos": records,
    }
    with manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(embedding_manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
