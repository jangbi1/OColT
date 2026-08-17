from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_INDEX = 999
CLASS_NAMES = (
    "ileum",
    "cecum",
    "ascending",
    "transverse",
    "descending",
    "sigmoid",
    "rectum",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
FRAME_RE = re.compile(r"_(\d+(?:\.\d+)?)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+)\s*$")


def parse_frame_number(name: str) -> int:
    match = FRAME_RE.search(Path(str(name).strip()).name)
    return -1 if match is None else int(float(match.group(1)))


def load_real_targets(
    csv_path: str | Path,
    image_names: list[str],
    label_column: str = "GT",
) -> np.ndarray:
    frame_to_label: dict[int, int] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "frame_filename" not in fields or label_column not in fields:
            raise ValueError(
                f"{csv_path} must contain frame_filename and {label_column}; found {fields}"
            )
        for row in reader:
            frame = parse_frame_number(row["frame_filename"])
            if frame < 0:
                raise ValueError(f"cannot parse frame_filename={row['frame_filename']!r}")
            if frame in frame_to_label:
                raise ValueError(f"duplicate REAL frame number {frame} in {csv_path}")
            label = str(row.get(label_column, "")).strip().lower()
            frame_to_label[frame] = CLASS_TO_INDEX.get(label, IGNORE_INDEX)

    target = np.full(len(image_names), IGNORE_INDEX, dtype=np.int64)
    for index, name in enumerate(image_names):
        frame = parse_frame_number(name)
        if frame >= 0:
            target[index] = frame_to_label.get(frame, IGNORE_INDEX)
    return target


def _seconds(value: str) -> int:
    match = TIME_RE.match(value)
    if match is None:
        raise ValueError(f"invalid CAS time value: {value!r}")
    return int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def _intervals(cell: Any) -> list[tuple[int, int]]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    result: list[tuple[int, int]] = []
    for part in (value.strip() for value in text.split("/")):
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"invalid CAS interval: {part!r}")
        start_text, end_text = (value.strip() for value in part.split("-", 1))
        start, end = _seconds(start_text), _seconds(end_text)
        if end < start:
            raise ValueError(f"CAS interval ends before it starts: {part!r}")
        result.append((start, end))
    return result


def load_cas_table(path: str | Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "VideoID" not in reader.fieldnames:
            raise ValueError(f"CAS label CSV lacks VideoID: {path}")
        for row in reader:
            video_id = str(row.get("VideoID", "")).strip()
            if not video_id:
                continue
            if video_id in result:
                raise ValueError(f"duplicate CAS VideoID {video_id!r}")
            result[video_id] = row
    return result


def build_cas_targets(
    row: Mapping[str, Any], length: int, fps: float = 5.0
) -> np.ndarray:

    columns = {
        "Terminal_Ileum": "ileum",
        "Cecum": "cecum",
        "Ascending_Colon": "ascending",
        "Hepatic_Flexure": "transverse",
        "Transverse_Colon": "transverse",
        "Splenic_Flexure": "transverse",
        "Descending_Colon": "descending",
        "Sigmoid_Colon": "sigmoid",
        "Rectum": "rectum",
        "Anal_Canal": "rectum",
    }
    target = np.full(length, IGNORE_INDEX, dtype=np.int64)
    for column, class_name in columns.items():
        for start_second, end_second in _intervals(row.get(column)):
            begin = max(0, min(length, int(math.floor(start_second * fps))))
            end = max(0, min(length, int(math.ceil((end_second + 1) * fps))))
            target[begin:end] = CLASS_TO_INDEX[class_name]
    return target
