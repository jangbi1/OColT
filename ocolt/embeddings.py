from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


EXPECTED_DIMENSION = 384


def validate_embedding_payload(
    payload: Mapping,
    *,
    expected_dimension: int = EXPECTED_DIMENSION,
) -> tuple[np.ndarray, list[str]]:
    if "video_embeddings" not in payload or "image_names" not in payload:
        raise ValueError("PKL must contain video_embeddings and image_names")
    embeddings = np.asarray(payload["video_embeddings"], dtype=np.float32)
    if embeddings.ndim == 2:
        embeddings = embeddings[np.newaxis, ...]
    if embeddings.ndim != 3:
        raise ValueError(
            f"video_embeddings must be (streams,T,D) or (T,D), got {embeddings.shape}"
        )
    if embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise ValueError("video_embeddings cannot be empty")
    if embeddings.shape[2] != expected_dimension:
        raise ValueError(
            f"expected embedding dimension {expected_dimension}, got {embeddings.shape[2]}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("video_embeddings contains NaN or infinity")

    raw_names = payload["image_names"]
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("image_names must be a sequence")
    image_names = [str(name) for name in raw_names]
    if len(image_names) != embeddings.shape[1]:
        raise ValueError(
            f"image_names length {len(image_names)} does not match T={embeddings.shape[1]}"
        )
    if len(set(image_names)) != len(image_names):
        raise ValueError("image_names contains duplicates")
    if any(not name.strip() for name in image_names):
        raise ValueError("image_names contains an empty value")
    return np.ascontiguousarray(embeddings), image_names


def load_embedding_pkl(path: str | Path) -> tuple[np.ndarray, list[str]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"embedding PKL root is not a mapping: {path}")
    return validate_embedding_payload(payload)


def save_embedding_pkl(
    path: str | Path,
    embeddings: np.ndarray,
    image_names: Sequence[str],
    metadata: Mapping | None = None,
) -> None:
    array, names = validate_embedding_payload(
        {"video_embeddings": embeddings, "image_names": list(image_names)}
    )
    payload: dict = {"video_embeddings": array, "image_names": names}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
