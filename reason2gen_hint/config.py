from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-2026-03-05")
DEFAULT_TIMEOUT = int(os.environ.get("GPT_API_TIMEOUT", 180))
DEFAULT_MAX_RETRIES = int(os.environ.get("GPT_API_MAX_RETRIES", 8))
MAX_DATAURL_CHARS = int(os.environ.get("MAX_DATAURL_CHARS", 1_200_000))
GRID_CELL_DEFAULT = 64
GRID_CELL_MIN = 48
GRID_CELL_MAX = 112
ROI_CELL_MIN = 22
ROI_CELL_MAX = 56
ROI_PAD_FRAC = 0.06


@dataclass
class RuntimeConfig:
    model: str = DEFAULT_MODEL
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    rpm: int = 600
    max_side_start: int = 1600
    mode: str = "single"
    refine_once: bool = False
    lite_save: bool = False
    grid_mode: str = "auto"
    grid_cell: int = GRID_CELL_DEFAULT
    out_root: str = "outputs"
    limit_rows: Optional[int] = None
