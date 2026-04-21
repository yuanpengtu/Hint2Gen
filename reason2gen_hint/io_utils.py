from __future__ import annotations

import base64
import io
import os
import re
from typing import Optional

from bs4 import BeautifulSoup, FeatureNotFound
from PIL import Image


def resize(img: Image.Image, max_side: int) -> Image.Image:
    if max_side is None or max_side <= 0:
        return img
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def bytes_from_cell(cell) -> bytes:
    if isinstance(cell, Image.Image):
        buf = io.BytesIO()
        cell.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(cell, str) and os.path.isfile(cell):
        with open(cell, "rb") as f:
            return f.read()
    if not isinstance(cell, dict):
        raise TypeError(f"Expect dict/path/image, got {type(cell)}")
    raw = cell.get("bytes")
    if raw is None:
        path = cell.get("path")
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
        else:
            raise ValueError("Missing 'bytes' and no readable 'path'")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    elif isinstance(raw, str):
        try:
            raw = base64.b64decode(raw)
        except Exception as exc:
            raise TypeError("String payload is not base64-encoded bytes") from exc
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"Not bytes: {type(raw)}")
    return bytes(raw)


def path_from_cell(cell) -> str:
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        return cell.get("path") or ""
    return ""


def extract_code_block(text: str, lang_hint: str = "html") -> str:
    m = re.search(rf"```{lang_hint}\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def soup(html: str):
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img


def safe_join(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def pick_first(record: dict, keys: list[str], default=None):
    for key in keys:
        if key in record and record[key] not in (None, "", []):
            return record[key]
    return default
