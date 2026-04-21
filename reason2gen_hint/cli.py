from __future__ import annotations

import argparse
import glob
import os
import sys

from loguru import logger

from .client import OpenAIChatClient
from .config import DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_TIMEOUT, GRID_CELL_DEFAULT, RuntimeConfig
from .datasets import rows_from_hf, rows_from_json, rows_from_parquet
from .pipeline import run_rows_parallel
from .rate_limit import RateLimiter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reason2Gen hint generation with official OpenAI API.")
    parser.add_argument("--dataset-source", choices=["huggingface", "json", "parquet"], default="huggingface")
    parser.add_argument("--hf-dataset", default="Tuyuanpeng/Reason2Gen")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--data-dir", default=None, help="Directory containing parquet shards.")
    parser.add_argument("--out-root", default="github_outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--rpm", type=int, default=600)
    parser.add_argument("--max-side-start", type=int, default=1600)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--refine-once", action="store_true")
    parser.add_argument("--lite-save", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--grid-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--grid-cell", type=int, default=GRID_CELL_DEFAULT)
    parser.add_argument("--api-keys", nargs="+", default=None, help="Official OpenAI API keys. If omitted, OPENAI_API_KEY is used.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    return parser


def load_rows(args) -> list:
    if args.dataset_source == "huggingface":
        return rows_from_hf(args.hf_dataset, split=args.hf_split, config_name=args.hf_config)
    if args.dataset_source == "json":
        if not args.json_path:
            raise ValueError("--json-path is required when --dataset-source json")
        return rows_from_json(args.json_path, args.image_root)
    if not args.data_dir:
        raise ValueError("--data-dir is required when --dataset-source parquet")
    rows = []
    for parquet_path in sorted(glob.glob(os.path.join(args.data_dir, "*.parquet"))):
        rows.extend(rows_from_parquet(parquet_path, image_root=args.image_root))
    return rows


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True)
    os.makedirs(args.out_root, exist_ok=True)

    api_keys = args.api_keys or ([os.environ["OPENAI_API_KEY"]] if os.environ.get("OPENAI_API_KEY") else None)
    if not api_keys:
        raise RuntimeError("Provide --api-keys or set OPENAI_API_KEY")

    cfg = RuntimeConfig(
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rpm=args.rpm,
        max_side_start=args.max_side_start,
        mode=args.mode,
        refine_once=args.refine_once,
        lite_save=args.lite_save,
        grid_mode=args.grid_mode,
        grid_cell=max(8, int(args.grid_cell)),
        out_root=args.out_root,
        limit_rows=args.limit_rows,
    )

    rows = load_rows(args)
    limiter = RateLimiter(rpm=cfg.rpm)
    clients = [
        OpenAIChatClient(
            api_key=api_key,
            limiter=limiter,
            model=cfg.model,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            base_url=args.base_url,
        )
        for api_key in api_keys
    ]
    total = run_rows_parallel(clients, rows, cfg, shard_name=args.dataset_source)
    logger.info(f"ALL DONE. Total processed rows: {total}")


if __name__ == "__main__":
    main()
