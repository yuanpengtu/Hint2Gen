from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from .io_utils import pick_first, safe_join


def load_json_records(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    try:
        obj = json.loads(f"[{txt.rstrip(',')}]")
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    items = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    if items:
        return items
    raise ValueError(f"Cannot parse JSON file: {json_path}")


def normalize_record(record: dict, image_root: str = "") -> Optional[SimpleNamespace]:
    prompt = pick_first(record, ["edit_instruction", "input_prompt", "instruction", "prompt"], "")
    image_id = str(pick_first(record, ["image_id", "id"], "") or "")
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    variant_dir = metadata.get("variant_dir", "") if metadata else ""

    original = pick_first(
        record,
        ["image_file", "input_image", "original_image", "source_image", "image", "image_input"],
    )
    edited = pick_first(
        record,
        ["edited_file", "output_image", "edited_image", "target_image", "image_target"],
    )

    if isinstance(original, list):
        original = original[0] if original else None
    if isinstance(edited, list):
        edited = edited[0] if edited else None
    if original is None or edited is None:
        return None

    def _resolve_path(value: str) -> dict:
        direct = safe_join(image_root, value)
        if os.path.isfile(direct):
            return {"path": direct}
        if variant_dir:
            nested = safe_join(image_root, os.path.join(variant_dir, value))
            if os.path.isfile(nested):
                return {"path": nested}
        return {"path": direct}

    if isinstance(original, str):
        original = _resolve_path(original)
    if isinstance(edited, str):
        edited = _resolve_path(edited)

    if not image_id:
        path = ""
        if isinstance(edited, dict):
            path = edited.get("path", "")
        if path:
            image_id = os.path.splitext(os.path.basename(path))[0]
        else:
            image_id = "sample"

    return SimpleNamespace(
        image_id=image_id,
        image_file=original,
        edited_file=edited,
        edit_instruction=prompt,
        metadata=metadata,
    )


def rows_from_json(json_path: str, image_root: str) -> List[SimpleNamespace]:
    rows = []
    for record in load_json_records(json_path):
        row = normalize_record(record, image_root=image_root)
        if row is not None:
            rows.append(row)
    return rows


def rows_from_benchmark_root(dataset_root: str) -> List[SimpleNamespace]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"dataset_root is not a directory: {dataset_root}")

    rows: List[SimpleNamespace] = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        json_candidates = sorted(subdir.glob("*.json"))
        if not json_candidates:
            continue
        question_dir = subdir / "question"
        answer_dir = subdir / "answer"
        if not question_dir.is_dir() or not answer_dir.is_dir():
            continue

        json_path = None
        named_json = subdir / f"{subdir.name}.json"
        if named_json.exists():
            json_path = named_json
        else:
            json_path = json_candidates[0]

        for record in load_json_records(str(json_path)):
            if not isinstance(record, dict):
                continue
            rec = dict(record)
            if "image_input" in rec and isinstance(rec["image_input"], str):
                rec["image_input"] = str(question_dir / rec["image_input"])
            if "image_target" in rec and isinstance(rec["image_target"], str):
                rec["image_target"] = str(answer_dir / rec["image_target"])
            row = normalize_record(rec, image_root="")
            if row is None:
                continue
            setattr(row, "category", subdir.name)
            rows.append(row)
    return rows


def rows_from_parquet(parquet_path: str, image_root: str = "") -> List[SimpleNamespace]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for parquet loading") from exc
    df = pd.read_parquet(parquet_path)
    resolved_image_root = image_root or str(Path(parquet_path).resolve().parent)
    rows = []
    for row in df.to_dict("records"):
        item = normalize_record(row, image_root=resolved_image_root)
        if item is not None:
            rows.append(item)
    return rows


def rows_from_hf(dataset_name: str, split: str = "train", config_name: Optional[str] = None) -> List[SimpleNamespace]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for Hugging Face dataset loading") from exc

    ds = load_dataset(dataset_name, name=config_name, split=split)
    rows = []
    for record in ds:
        row = normalize_record(dict(record))
        if row is not None:
            rows.append(row)
    return rows
