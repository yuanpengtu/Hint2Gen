from __future__ import annotations

import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from openai import BadRequestError
from PIL import Image

from .client import OpenAIChatClient
from .config import RuntimeConfig
from .io_utils import bytes_from_cell, encode_jpeg, extract_code_block, path_from_cell, to_data_url
from .prompts import build_multi_pass_messages, build_refine_messages, build_single_pass_messages
from .svg_ops import (
    draw_shapes_on_image,
    html_from_shapes,
    parse_svg_shapes,
    rasterize_mask,
    scale_shapes,
    strip_fills_and_force_strokes,
)
from .vision import diff_mask, make_adaptive_payload, make_diff_precise, make_edges


def gpt_chat(client: OpenAIChatClient, messages: List[dict], max_tokens: int = 8192) -> str:
    return client.chat(messages, max_tokens=max_tokens)


def render_shapes_rgba(size: Tuple[int, int], shapes: List[dict]) -> Image.Image:
    w, h = size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_shapes_on_image(canvas, shapes, out_png_path="/tmp/__unused.png")
    return canvas


def critique_from_masks(svg_mask: np.ndarray, diff_mask_arr: np.ndarray) -> Tuple[str, Image.Image]:
    h, w = diff_mask_arr.shape
    miss = np.logical_and(diff_mask_arr > 0, svg_mask == 0)
    over = np.logical_and(svg_mask > 0, diff_mask_arr == 0)
    critique = (
        "Refine instructions:\n"
        f"- Missing coverage (cyan) is about {100.0 * miss.sum() / float(h * w):.1f}% of the canvas.\n"
        f"- Over-coverage (orange) is about {100.0 * over.sum() / float(h * w):.1f}% of the canvas.\n"
        "- Tighten to true edges and remove unchanged background.\n"
        "- Add missing local structures visible in EDITED.\n"
    )
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[miss] = (0, 220, 220)
    overlay[over] = (255, 160, 0)
    return critique, Image.fromarray(overlay, mode="RGB")


def diff_fallback(original: Image.Image, edited: Image.Image, out_png_path: str):
    original.convert("RGB").save(out_png_path)


def gen_html_single_pass(
    client: OpenAIChatClient,
    o_img: Image.Image,
    e_img: Image.Image,
    cfg: RuntimeConfig,
    edit_instruction: Optional[str],
) -> Tuple[str, Tuple[int, int]]:
    du_o, du_e, (w, h) = make_adaptive_payload(
        o_img, e_img, max_side_start=cfg.max_side_start, with_grid=True, grid_mode=cfg.grid_mode, grid_cell=cfg.grid_cell
    )
    _, diff_img, e_aligned = make_diff_precise(o_img, e_img, (w, h))
    edges_img = make_edges(e_aligned, (w, h))
    msgs = build_single_pass_messages(
        du_o,
        du_e,
        to_data_url(encode_jpeg(diff_img, 85)),
        to_data_url(encode_jpeg(edges_img, 85)),
        w,
        h,
        edit_instruction,
    )
    raw = gpt_chat(client, msgs, max_tokens=8192)
    html = extract_code_block(raw, "html")
    if "<img" in html and "src=" in html:
        html = re.sub(r'<img([^>]+)src=["\'](.*?)["\']', lambda m: f'<img{m.group(1)}src="{du_o}"', html, flags=re.I)
    else:
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hints</title></head>
<body><div class="wrap"><img src="{du_o}" width="{w}" height="{h}" /><svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg></div></body></html>"""
    return strip_fills_and_force_strokes(html), (w, h)


def gen_html_multi_pass(
    client: OpenAIChatClient,
    o_img: Image.Image,
    e_img: Image.Image,
    cfg: RuntimeConfig,
    edit_instruction: Optional[str],
) -> Tuple[str, Tuple[int, int], str, str, str, str]:
    du_o, du_e, (w, h) = make_adaptive_payload(
        o_img, e_img, max_side_start=cfg.max_side_start, with_grid=True, grid_mode=cfg.grid_mode, grid_cell=cfg.grid_cell
    )
    _, diff_img, e_aligned = make_diff_precise(o_img, e_img, (w, h))
    edges_img = make_edges(e_aligned, (w, h))
    du_diff = to_data_url(encode_jpeg(diff_img, 85))
    du_edges = to_data_url(encode_jpeg(edges_img, 85))
    msgs = build_multi_pass_messages(
        du_o,
        du_e,
        du_diff,
        du_edges,
        w,
        h,
        edit_instruction,
    )
    raw = gpt_chat(client, msgs, max_tokens=8192)
    html = extract_code_block(raw, "html")
    if "<img" in html and "src=" in html:
        html = re.sub(r'<img([^>]+)src=["\'](.*?)["\']', lambda m: f'<img{m.group(1)}src="{du_o}"', html, flags=re.I)
    else:
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hints</title></head>
<body><div class="wrap"><img src="{du_o}" width="{w}" height="{h}" /><svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg></div></body></html>"""
    return strip_fills_and_force_strokes(html), (w, h), du_o, du_e, du_diff, du_edges


def refine_single_pass_once(
    client: OpenAIChatClient,
    o_img: Image.Image,
    e_img: Image.Image,
    html_first: str,
    canvas_wh: Tuple[int, int],
    cfg: RuntimeConfig,
    edit_instruction: Optional[str],
) -> str:
    shapes_first, _ = parse_svg_shapes(html_first)
    w, h = canvas_wh
    du_o, du_e, _ = make_adaptive_payload(
        o_img, e_img, max_side_start=max(w, h), with_grid=True, grid_mode=cfg.grid_mode, grid_cell=cfg.grid_cell
    )
    _, diff_img, e_aligned = make_diff_precise(o_img, e_img, (w, h))
    edges_img = make_edges(e_aligned, (w, h))
    critique, overlay_rgb = critique_from_masks(
        rasterize_mask(shapes_first, (w, h)),
        diff_mask(o_img.resize((w, h), Image.LANCZOS), e_img.resize((w, h), Image.LANCZOS)),
    )
    base = Image.new("RGB", (w, h), (0, 0, 0))
    base.paste(overlay_rgb, (0, 0))
    du_overlay = to_data_url(encode_jpeg(base, 85))
    msgs = build_refine_messages(
        du_o,
        du_e,
        to_data_url(encode_jpeg(diff_img, 85)),
        to_data_url(encode_jpeg(edges_img, 85)),
        du_overlay,
        w,
        h,
        critique,
        edit_instruction,
    )
    raw = gpt_chat(client, msgs, max_tokens=8192)
    html = extract_code_block(raw, "html")
    if "<img" in html and "src=" in html:
        html = re.sub(r'<img([^>]+)src=["\'](.*?)["\']', lambda m: f'<img{m.group(1)}src="{du_o}"', html, flags=re.I)
    return strip_fills_and_force_strokes(html)


def process_row(client: OpenAIChatClient, row, cfg: RuntimeConfig, out_dir: str, idx_tag: str) -> None:
    sid_raw = str(getattr(row, "image_id", "")) or idx_tag
    sid = re.sub(r"[^A-Za-z0-9._-]+", "_", sid_raw)
    png_path = os.path.join(out_dir, f"hints_{sid}.png")
    if os.path.exists(png_path):
        logger.info(f"Skip existing hint PNG: {png_path}")
        return

    o_bytes = bytes_from_cell(row.image_file)
    e_bytes = bytes_from_cell(row.edited_file)
    logger.info(f"[{sid}] source_path={path_from_cell(row.image_file)}")
    logger.info(f"[{sid}] target_path={path_from_cell(row.edited_file)}")
    o_img_orig = Image.open(io.BytesIO(o_bytes)).convert("RGBA")
    e_img_orig = Image.open(io.BytesIO(e_bytes)).convert("RGBA")
    if o_img_orig.size != e_img_orig.size:
        e_img_orig = e_img_orig.resize(o_img_orig.size, Image.LANCZOS)
    edit_instruction = getattr(row, "edit_instruction", None)
    w0, h0 = o_img_orig.size

    if cfg.mode == "single":
        html_model, (wc, hc) = gen_html_single_pass(client, o_img_orig, e_img_orig, cfg, edit_instruction)
        if cfg.refine_once:
            try:
                html_model = refine_single_pass_once(client, o_img_orig, e_img_orig, html_model, (wc, hc), cfg, edit_instruction)
            except Exception as exc:
                logger.warning(f"refine-once skipped due to error: {exc}")
    else:
        html_model, (wc, hc), du_o, du_e, du_diff, du_edges = gen_html_multi_pass(
            client, o_img_orig, e_img_orig, cfg, edit_instruction
        )
        shapes_current, _ = parse_svg_shapes(html_model)
        o_for_q = o_img_orig.resize((wc, hc), Image.LANCZOS)
        e_for_q = e_img_orig.resize((wc, hc), Image.LANCZOS)

        def quality(shapes: List[dict]) -> Dict[str, float]:
            if not shapes:
                return {"coverage": 0.0, "avg_vertices": 0.0, "polygons": 0.0, "iou": 0.0}
            svg = rasterize_mask(shapes, (wc, hc))
            dif = diff_mask(o_for_q, e_for_q)
            polys = [s for s in shapes if s.get("type") == "polygon"]
            verts = [len(s.get("points", [])) for s in polys]
            inter = np.logical_and(svg > 0, dif > 0).sum()
            union = np.logical_or(svg > 0, dif > 0).sum()
            return {
                "coverage": float(svg.sum()) / float(255 * wc * hc),
                "avg_vertices": float(sum(verts) / max(1, len(verts))) if verts else 0.0,
                "polygons": float(len(polys)),
                "iou": float(inter) / float(union) if union > 0 else 0.0,
            }

        q = quality(shapes_current)
        rounds = 1
        while rounds < 3 and not (q["iou"] >= 0.45 and 0.02 <= q["coverage"] <= 0.40 and q["polygons"] >= 3):
            critique = (
                "Refine the hint: increase local detail, remove coarse false positives, "
                f"and improve IoU against the changed region. Current IoU={q['iou']:.3f}, coverage={q['coverage']:.3f}."
            )
            overlay_prev = render_shapes_rgba((wc, hc), shapes_current).convert("RGB")
            du_overlay_prev = to_data_url(encode_jpeg(overlay_prev, 85))
            html_new = extract_code_block(
                gpt_chat(
                    client,
                    build_refine_messages(
                        du_o,
                        du_e,
                        du_diff,
                        du_edges,
                        du_overlay_prev,
                        wc,
                        hc,
                        critique,
                        edit_instruction,
                    ),
                    max_tokens=8192,
                ),
                "html",
            )
            shapes_new, _ = parse_svg_shapes(html_new)
            q_new = quality(shapes_new)
            if q_new["iou"] > q["iou"]:
                html_model = html_new
                shapes_current = shapes_new
                q = q_new
            rounds += 1

    shapes_c, _ = parse_svg_shapes(html_model)
    scaled_shapes = scale_shapes(shapes_c, w0 / float(max(1, wc)), h0 / float(max(1, hc))) if wc > 0 and hc > 0 else []
    if not cfg.lite_save:
        with open(os.path.join(out_dir, f"hints_{sid}.html"), "w", encoding="utf-8") as f:
            f.write(html_model)
    du_orig = to_data_url(encode_jpeg(o_img_orig.convert("RGB"), 90))
    with open(os.path.join(out_dir, f"hints_{sid}_orig.html"), "w", encoding="utf-8") as f:
        f.write(html_from_shapes(du_orig, w0, h0, scaled_shapes))
    if scaled_shapes:
        draw_shapes_on_image(o_img_orig, scaled_shapes, png_path)
    else:
        diff_fallback(o_img_orig, e_img_orig, png_path)


def run_rows(client: OpenAIChatClient, rows: List, cfg: RuntimeConfig, shard_name: str = "default") -> int:
    out_dir = os.path.join(cfg.out_root, shard_name)
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for idx, row in enumerate(rows):
        if cfg.limit_rows is not None and total >= cfg.limit_rows:
            break
        try:
            process_row(client, row, cfg, out_dir, idx_tag=f"{shard_name}_{idx}")
            total += 1
        except BadRequestError as exc:
            logger.error(f"[{shard_name} @ row {idx}] 400: {exc}")
        except Exception as exc:
            logger.error(f"[{shard_name} @ row {idx}] failed: {exc}")
    return total


def run_rows_parallel(clients: List[OpenAIChatClient], rows: List, cfg: RuntimeConfig, shard_name: str = "default") -> int:
    n_workers = max(1, len(clients))
    slices = [rows[i::n_workers] for i in range(n_workers)]
    total = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_rows, client, rows_slice, cfg, f"{shard_name}_{i}") for i, (client, rows_slice) in enumerate(zip(clients, slices))]
        for fut in as_completed(futs):
            total += fut.result()
    return total
