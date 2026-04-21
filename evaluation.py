from __future__ import annotations

import argparse
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from loguru import logger
from openai import OpenAI
from PIL import Image

from reason2gen_hint.datasets import normalize_record, rows_from_benchmark_root
from reason2gen_hint.io_utils import bytes_from_cell, encode_jpeg, to_data_url


DEFAULT_EVAL_MODEL = os.environ.get("OPENAI_EVAL_MODEL", "gpt-5.4-2026-03-05")
DEFAULT_GEMINI_EVAL_MODEL = os.environ.get("GEMINI_EVAL_MODEL", "gemini-2.5-pro")

JUDGE_SYSTEM_PROMPT = """You are the evaluator for a reasoning-aware image generation benchmark.

You will receive:
1. the question image (or problem image),
2. the candidate generated image,
3. the ground-truth target image,
4. the original question or edit instruction.

Evaluate the candidate according to the paper protocol:
- Dimension 1: consistency with the question / edit instruction
- Dimension 2: correctness relative to the expected visual outcome

Scoring rule:
- score = 1 only if both dimensions are satisfied
- score = 0 otherwise

Return compact JSON only with this schema:
{
  "instruction_consistency": 0 or 1,
  "target_correctness": 0 or 1,
  "score": 0 or 1,
  "reason": "short explanation"
}

Be strict. Focus on logical correctness, not surface-level similarity alone.
"""


@dataclass
class EvalSample:
    image_id: str
    prompt: str
    question_bytes: bytes
    target_bytes: bytes
    category: str = ""
    reasoning_dimension: str = ""
    question_name: str = ""
    target_name: str = ""
    numeric_key: str = ""


JUDGE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction_consistency": {"type": "integer", "enum": [0, 1]},
        "target_correctness": {"type": "integer", "enum": [0, 1]},
        "score": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string"},
    },
    "required": [
        "instruction_consistency",
        "target_correctness",
        "score",
        "reason",
    ],
}


DIMENSION_TO_TASK_ALIASES: Dict[str, List[str]] = {
    "path_connectivity_reasoning": [
        "waterpipe connection",
        "waterpipe",
        "water_pipe",
        "pipe",
        "box moving",
        "box_moving",
        "boxmove",
        "sokoban",
        "ball trajectory prediction",
        "ball trajectory",
        "ball_trajectory",
        "trajectory",
        "ballbounce",
        "maze solving",
        "maze",
        "maze_path",
        "car parking",
        "car_parking",
        "parking",
        "carpark",
    ],
    "spatial_assembly_geometric_imagination": [
        "tangram",
        "tangram assembly",
        "jigsaw",
        "spatial perspective generation",
        "spatial perspective",
        "spatial_perspective",
        "perspective",
        "spatial",
        "draw the net",
        "net",
        "cube net",
        "unfold",
    ],
    "rule_based_pattern_induction": [
        "visual logic puzzles",
        "visual logic",
        "visual_logic",
        "logic puzzle",
        "puzzle",
        "clock-time reasoning",
        "clock time reasoning",
        "clock",
        "clock_time",
        "reasoning connection",
        "reasoning_connection",
        "connection",
        "connect-the-dots",
        "connect the dots",
        "connection matching",
        "line",
    ],
    "detail_comparison_visual_retrieval": [
        "spot-the-difference",
        "spot the difference",
        "spot_difference",
        "difference",
        "disappearance detection",
        "disappearance",
        "missing object",
        "missing",
    ],
    "language_commonsense_reasoning": [
        "word-search puzzles",
        "word-searching",
        "word search",
        "word_search",
        "wordsearch",
        "crossword",
        "sudoku",
        "sudo",
    ],
    "rapid_pattern_matching_elimination": [
        "tetris",
        "zuma-style elimination",
        "zuma style elimination",
        "zuma",
    ],
    "strategic_planning_long_horizon_reasoning": [
        "light placement",
        "light placement puzzle",
        "light_up",
        "lightup",
        "tower of hanoi",
        "hanoi",
        "gobang",
        "gomoku",
        "chess",
        "international chess",
        "chess_dataset",
        "chinese chess",
        "xiangqi",
        "chinachess",
        "klotski",
        "huarong road",
        "huarong",
        "light",
    ],
}


DIMENSION_DISPLAY_NAMES: Dict[str, str] = {
    "path_connectivity_reasoning": "Path & Connectivity Reasoning",
    "spatial_assembly_geometric_imagination": "Spatial Assembly & Geometric Imagination",
    "rule_based_pattern_induction": "Rule-Based Pattern Induction",
    "detail_comparison_visual_retrieval": "Detail Comparison & Visual Retrieval",
    "language_commonsense_reasoning": "Language & Commonsense Reasoning",
    "rapid_pattern_matching_elimination": "Rapid Pattern Matching & Elimination",
    "strategic_planning_long_horizon_reasoning": "Strategic Planning with Long-Horizon Reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Reason2Gen-style generation results using GPT-5.4 as judge.")
    parser.add_argument("--judge-backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--dataset-source", choices=["huggingface", "json", "benchmark_root"], default="huggingface")
    parser.add_argument("--hf-dataset", default="Tuyuanpeng/Reason2Gen")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--dataset-root", default=None, help="Benchmark root directory containing 22 subfolders. Each subfolder should contain question/, answer/, and a <name>.json file.")
    parser.add_argument(
        "--pred-roots",
        nargs="+",
        required=True,
        help="One or more directories containing generated images. If multiple roots are provided, the script reports each run and the mean accuracy across runs.",
    )
    parser.add_argument(
        "--pred-pattern",
        default="{image_id}.png",
        help="Filename pattern under each prediction root. Example: '{image_id}.png' or 'result_{image_id}.jpg'.",
    )
    parser.add_argument(
        "--pred-tag",
        default=None,
        help="Optional generation tag used by baselines that save outputs as <numeric_key>_<tag>.png under <pred_root>/<category>/<tag>/.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save detailed evaluation results as JSON.")
    parser.add_argument("--model", default=None, help="Judge model name. Defaults depend on --judge-backend.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"), help="Google Gemini API key.")
    parser.add_argument("--gemini-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), help="Optional Google Cloud project for Vertex AI mode.")
    parser.add_argument("--gemini-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION"), help="Optional Google Cloud location for Vertex AI mode.")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


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
    lines = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    if lines:
        return lines
    raise ValueError(f"Cannot parse JSON file: {json_path}")


def extract_numeric_key(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(\d+)", stem)
    if m:
        return m.group(1)
    return stem


def normalize_name(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\bdataset\b", " ", text)
    text = re.sub(r"\bperfect\b", " ", text)
    text = re.sub(r"\btrain\b", " ", text)
    text = re.sub(r"\btest\b", " ", text)
    text = re.sub(r"\bval\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_reasoning_dimension(category: str) -> Optional[str]:
    name = normalize_name(category)
    if not name:
        return None
    for dimension, aliases in DIMENSION_TO_TASK_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_name(alias)
            if name == alias_norm or alias_norm in name or name in alias_norm:
                return dimension
    return None


def load_samples(args: argparse.Namespace) -> List[EvalSample]:
    if args.dataset_source == "huggingface":
        ds = load_dataset(args.hf_dataset, name=args.hf_config, split=args.hf_split)
        records = [dict(item) for item in ds]
    elif args.dataset_source == "json":
        if not args.json_path:
            raise ValueError("--json-path is required when --dataset-source json")
        records = load_json_records(args.json_path)
    else:
        if not args.dataset_root:
            raise ValueError("--dataset-root is required when --dataset-source benchmark_root")
        rows = rows_from_benchmark_root(args.dataset_root)
        samples: List[EvalSample] = []
        for row in rows:
            image_id = str(getattr(row, "image_id", "") or "")
            prompt = str(getattr(row, "edit_instruction", "") or "")
            if not image_id:
                continue
            image_file = getattr(row, "image_file", {})
            edited_file = getattr(row, "edited_file", {})
            question_name = os.path.basename(image_file.get("path", "")) if isinstance(image_file, dict) else ""
            target_name = os.path.basename(edited_file.get("path", "")) if isinstance(edited_file, dict) else ""
            samples.append(
                EvalSample(
                    image_id=image_id,
                    prompt=prompt,
                    question_bytes=bytes_from_cell(image_file),
                    target_bytes=bytes_from_cell(edited_file),
                    category=str(getattr(row, "category", "") or ""),
                    reasoning_dimension=infer_reasoning_dimension(str(getattr(row, "category", "") or "")) or "",
                    question_name=question_name,
                    target_name=target_name,
                    numeric_key=extract_numeric_key(question_name or image_id),
                )
            )
            if args.max_samples is not None and len(samples) >= args.max_samples:
                break
        return samples

    samples: List[EvalSample] = []
    for record in records:
        row = normalize_record(record, image_root=args.image_root)
        if row is None:
            continue
        image_id = str(getattr(row, "image_id", "") or "")
        prompt = str(getattr(row, "edit_instruction", "") or "")
        if not image_id:
            continue
        image_file = getattr(row, "image_file", {})
        edited_file = getattr(row, "edited_file", {})
        question_bytes = bytes_from_cell(row.image_file)
        target_bytes = bytes_from_cell(row.edited_file)
        samples.append(
            EvalSample(
                image_id=image_id,
                prompt=prompt,
                question_bytes=question_bytes,
                target_bytes=target_bytes,
                category=str(getattr(row, "category", "") or ""),
                reasoning_dimension=infer_reasoning_dimension(str(getattr(row, "category", "") or "")) or "",
                question_name=os.path.basename(image_file.get("path", "")) if isinstance(image_file, dict) else "",
                target_name=os.path.basename(edited_file.get("path", "")) if isinstance(edited_file, dict) else "",
                numeric_key=extract_numeric_key(
                    os.path.basename(image_file.get("path", "")) if isinstance(image_file, dict) else image_id
                ),
            )
        )
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break
    return samples


def resolve_prediction_path(pred_root: str, sample: EvalSample, pattern: str, pred_tag: Optional[str] = None) -> Path:
    root = Path(pred_root)
    candidates = []

    fmt = {
        "image_id": sample.image_id,
        "category": sample.category,
        "question_name": sample.question_name,
        "target_name": sample.target_name,
        "question_stem": Path(sample.question_name).stem if sample.question_name else "",
        "target_stem": Path(sample.target_name).stem if sample.target_name else "",
        "numeric_key": sample.numeric_key,
        "tag": pred_tag or "",
    }
    try:
        candidates.append(root / pattern.format(**fmt))
    except KeyError:
        candidates.append(root / pattern.format(image_id=sample.image_id))

    if sample.category:
        candidates.append(root / sample.category / pattern.format(image_id=sample.image_id))

    if pred_tag and sample.numeric_key:
        candidates.append(root / sample.category / pred_tag / f"{sample.numeric_key}_{pred_tag}.png" if sample.category else root / pred_tag / f"{sample.numeric_key}_{pred_tag}.png")
        candidates.append(root / f"{sample.numeric_key}_{pred_tag}.png")

    if sample.question_name:
        candidates.append(root / sample.question_name)
        candidates.append(root / sample.category / sample.question_name if sample.category else root / sample.question_name)
    if sample.target_name:
        candidates.append(root / sample.target_name)
        candidates.append(root / sample.category / sample.target_name if sample.category else root / sample.target_name)

    stems = [sample.image_id, sample.numeric_key]
    if sample.question_name:
        stems.append(Path(sample.question_name).stem)
    if sample.target_name:
        stems.append(Path(sample.target_name).stem)
    for stem in stems:
        if not stem:
            continue
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidates.append(root / f"{stem}{ext}")
            if sample.category:
                candidates.append(root / sample.category / f"{stem}{ext}")

    seen = set()
    for path in candidates:
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists():
            return path
    if pred_tag:
        recursive_hits = sorted(root.glob(f"**/{sample.numeric_key}_{pred_tag}.png"))
        if recursive_hits:
            return recursive_hits[0]
    recursive_generic = []
    for stem in filter(None, stems):
        recursive_generic.extend(sorted(root.glob(f"**/{stem}.png")))
        recursive_generic.extend(sorted(root.glob(f"**/{stem}.jpg")))
        recursive_generic.extend(sorted(root.glob(f"**/{stem}.jpeg")))
        recursive_generic.extend(sorted(root.glob(f"**/{stem}.webp")))
    if recursive_generic:
        return recursive_generic[0]
    return candidates[0]


def image_bytes_to_data_url(raw: bytes) -> str:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return to_data_url(encode_jpeg(pil, 90))


def detect_image_mime(raw: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = (img.format or "").upper()
    except Exception:
        fmt = ""
    return {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }.get(fmt, "image/png")


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def normalize_judge_result(obj: Optional[dict], raw_text: str) -> dict:
    obj = obj or {}
    inst = int(obj.get("instruction_consistency", 0))
    corr = int(obj.get("target_correctness", 0))
    score = int(obj.get("score", 1 if (inst == 1 and corr == 1) else 0))
    if score not in (0, 1):
        score = 1 if (inst == 1 and corr == 1) else 0
    return {
        "instruction_consistency": 1 if inst == 1 else 0,
        "target_correctness": 1 if corr == 1 else 0,
        "score": score,
        "reason": str(obj.get("reason", "")).strip(),
        "raw_response": raw_text,
    }


def judge_sample_openai(client: OpenAI, model: str, sample: EvalSample, pred_bytes: bytes) -> dict:
    q_url = image_bytes_to_data_url(sample.question_bytes)
    p_url = image_bytes_to_data_url(pred_bytes)
    t_url = image_bytes_to_data_url(sample.target_bytes)
    user_prompt = (
        "Evaluate this sample for the Reason2Gen / Hint2Gen benchmark.\n"
        "Judge whether the candidate image correctly solves the visual reasoning task.\n\n"
        f"Question / edit instruction:\n{sample.prompt}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "text", "text": "Question image"},
                    {"type": "image_url", "image_url": {"url": q_url}},
                    {"type": "text", "text": "Candidate generated image"},
                    {"type": "image_url", "image_url": {"url": p_url}},
                    {"type": "text", "text": "Ground-truth target image"},
                    {"type": "image_url", "image_url": {"url": t_url}},
                ],
            },
        ],
        max_tokens=500,
    )
    content = response.choices[0].message.content or ""
    return normalize_judge_result(extract_json_object(content), content)


def judge_sample_gemini(client: Any, model: str, sample: EvalSample, pred_bytes: bytes) -> dict:
    try:
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("google-genai is required for --judge-backend gemini") from exc

    response = client.models.generate_content(
        model=model,
        contents=[
            (
                "Evaluate this sample for the Reason2Gen / Hint2Gen benchmark.\n"
                "Judge whether the candidate image correctly solves the visual reasoning task.\n\n"
                f"Question / edit instruction:\n{sample.prompt}\n\n"
                "You must compare the candidate image against both the question image and the ground-truth target image."
            ),
            "Question image",
            types.Part.from_bytes(data=sample.question_bytes, mime_type=detect_image_mime(sample.question_bytes)),
            "Candidate generated image",
            types.Part.from_bytes(data=pred_bytes, mime_type=detect_image_mime(pred_bytes)),
            "Ground-truth target image",
            types.Part.from_bytes(data=sample.target_bytes, mime_type=detect_image_mime(sample.target_bytes)),
        ],
        config={
            "temperature": 0,
            "system_instruction": JUDGE_SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": JUDGE_RESPONSE_SCHEMA,
        },
    )
    content = getattr(response, "text", "") or ""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return normalize_judge_result(parsed, content)
    return normalize_judge_result(extract_json_object(content), content)


def build_judge_client(args: argparse.Namespace) -> tuple[str, Any]:
    if args.judge_backend == "openai":
        if not args.api_key:
            raise RuntimeError("Provide --api-key or set OPENAI_API_KEY for --judge-backend openai")
        model = args.model or DEFAULT_EVAL_MODEL
        return model, OpenAI(api_key=args.api_key, base_url=args.base_url)

    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is required for --judge-backend gemini") from exc

    if not args.gemini_api_key:
        raise RuntimeError("Provide --gemini-api-key or set GEMINI_API_KEY for --judge-backend gemini")

    model = args.model or DEFAULT_GEMINI_EVAL_MODEL
    client_kwargs: Dict[str, Any] = {"api_key": args.gemini_api_key}
    if args.gemini_project and args.gemini_location:
        client_kwargs["vertexai"] = True
        client_kwargs["project"] = args.gemini_project
        client_kwargs["location"] = args.gemini_location
    return model, genai.Client(**client_kwargs)


def evaluate_run(
    client: Any,
    judge_backend: str,
    model: str,
    samples: List[EvalSample],
    pred_root: str,
    pred_pattern: str,
    pred_tag: Optional[str] = None,
) -> dict:
    results = []
    missing = 0
    for idx, sample in enumerate(samples, 1):
        pred_path = resolve_prediction_path(pred_root, sample, pred_pattern, pred_tag=pred_tag)
        if not pred_path.exists():
            logger.warning(f"missing prediction: {pred_path}")
            missing += 1
            results.append(
                {
                    "image_id": sample.image_id,
                    "category": sample.category,
                    "reasoning_dimension": sample.reasoning_dimension,
                    "prompt": sample.prompt,
                    "prediction_path": str(pred_path),
                    "missing_prediction": True,
                    "instruction_consistency": 0,
                    "target_correctness": 0,
                    "score": 0,
                    "reason": "missing prediction",
                }
            )
            continue
        with open(pred_path, "rb") as f:
            pred_bytes = f.read()
        if judge_backend == "gemini":
            judge = judge_sample_gemini(client, model, sample, pred_bytes)
        else:
            judge = judge_sample_openai(client, model, sample, pred_bytes)
        results.append(
            {
                "image_id": sample.image_id,
                "category": sample.category,
                "reasoning_dimension": sample.reasoning_dimension,
                "prompt": sample.prompt,
                "prediction_path": str(pred_path),
                "missing_prediction": False,
                **judge,
            }
        )
        if idx % 20 == 0:
            logger.info(f"[{pred_root}] evaluated {idx}/{len(samples)}")

    scores = [item["score"] for item in results]
    accuracy = float(sum(scores)) / float(len(scores)) if scores else 0.0
    category_scores: Dict[str, List[int]] = {}
    for item in results:
        cat = item.get("category", "") or "unknown"
        category_scores.setdefault(cat, []).append(int(item["score"]))
    category_accuracy = {
        cat: (float(sum(vals)) / float(len(vals)) if vals else 0.0)
        for cat, vals in sorted(category_scores.items())
    }
    dimension_scores: Dict[str, List[int]] = {}
    unmapped_categories = set()
    for item in results:
        dim = item.get("reasoning_dimension", "") or ""
        cat = item.get("category", "") or "unknown"
        if not dim:
            unmapped_categories.add(cat)
            dim = "unmapped"
        dimension_scores.setdefault(dim, []).append(int(item["score"]))
    dimension_accuracy = {
        DIMENSION_DISPLAY_NAMES.get(dim, dim): (float(sum(vals)) / float(len(vals)) if vals else 0.0)
        for dim, vals in sorted(dimension_scores.items())
    }
    return {
        "pred_root": pred_root,
        "num_samples": len(results),
        "missing_predictions": missing,
        "accuracy": accuracy,
        "category_accuracy": category_accuracy,
        "dimension_accuracy": dimension_accuracy,
        "unmapped_categories": sorted(unmapped_categories),
        "results": results,
    }


def main():
    args = parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    samples = load_samples(args)
    logger.info(f"Loaded {len(samples)} evaluation samples.")

    model, client = build_judge_client(args)
    logger.info(f"Judge backend={args.judge_backend}, model={model}")
    run_reports = []
    for pred_root in args.pred_roots:
        report = evaluate_run(
            client,
            args.judge_backend,
            model,
            samples,
            pred_root,
            args.pred_pattern,
            pred_tag=args.pred_tag,
        )
        run_reports.append(report)
        logger.info(
            f"run={pred_root} accuracy={report['accuracy']:.4f} "
            f"samples={report['num_samples']} missing={report['missing_predictions']}"
        )

    mean_accuracy = mean([item["accuracy"] for item in run_reports]) if run_reports else 0.0
    summary = {
        "judge_backend": args.judge_backend,
        "model": model,
        "dataset_source": args.dataset_source,
        "hf_dataset": args.hf_dataset if args.dataset_source == "huggingface" else None,
        "hf_split": args.hf_split if args.dataset_source == "huggingface" else None,
        "num_runs": len(run_reports),
        "mean_accuracy": mean_accuracy,
        "runs": run_reports,
    }

    print("\n=== Evaluation Summary ===")
    for report in run_reports:
        print(f"{report['pred_root']}: accuracy={report['accuracy']:.4f}")
        if report.get("dimension_accuracy"):
            for dim, acc in report["dimension_accuracy"].items():
                print(f"  [dimension] {dim}: {acc:.4f}")
        if report.get("category_accuracy"):
            for cat, acc in report["category_accuracy"].items():
                print(f"  {cat}: {acc:.4f}")
        if report.get("unmapped_categories"):
            print(f"  unmapped_categories={report['unmapped_categories']}")
    print(f"mean_accuracy={mean_accuracy:.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved detailed report to {out_path}")


if __name__ == "__main__":
    main()
