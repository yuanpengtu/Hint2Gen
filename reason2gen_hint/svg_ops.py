from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io_utils import ensure_rgba, soup


NAMED = {
    "red": "#ff0000",
    "blue": "#0000ff",
    "green": "#00ff00",
    "lime": "#00ff88",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "orange": "#ff9500",
    "white": "#ffffff",
    "black": "#000000",
}


def norm_hex(color: str) -> Optional[str]:
    if not color:
        return None
    color = color.strip().lower()
    if color in NAMED:
        return NAMED[color]
    m = re.match(r"#([0-9a-f]{3})$", color)
    if m:
        s = m.group(1)
        return "#" + "".join(ch * 2 for ch in s)
    m = re.match(r"#([0-9a-f]{6})$", color)
    if m:
        return "#" + m.group(1)
    return None


def rgba_from_css(color: Optional[str], alpha: int = 255) -> Tuple[int, int, int, int]:
    color = norm_hex(color or "")
    if not color:
        return (255, 77, 79, alpha)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return (r, g, b, alpha)


def pick_style_color(tag) -> Optional[str]:
    color = tag.get("stroke") or tag.get("fill")
    style = tag.get("style", "")
    if not color and "stroke:" in style:
        m = re.search(r"stroke\s*:\s*([^;]+)", style, flags=re.I)
        color = m.group(1) if m else None
    if not color and "fill:" in style and tag.name == "text":
        m = re.search(r"fill\s*:\s*([^;]+)", style, flags=re.I)
        color = m.group(1) if m else None
    return norm_hex(color) if color else None


def pick_style_stroke_width(tag) -> Optional[float]:
    sw = tag.get("stroke-width")
    if not sw:
        m = re.search(r"stroke-width\s*:\s*([0-9.]+)", tag.get("style", ""), flags=re.I)
        sw = m.group(1) if m else None
    try:
        return float(sw) if sw else None
    except Exception:
        return None


def strip_fills_and_force_strokes(html: str) -> str:
    html = re.sub(r'(<(?:polygon|polyline|rect|circle)\b[^>]*?)\s+fill\s*=\s*"[^"]*"', r"\1", html, flags=re.I)
    html = re.sub(r"(<(?:polygon|polyline|rect|circle)\b[^>]*?)>", r'\1 fill="none">', html, flags=re.I)
    if "</head>" in html:
        html = html.replace(
            "</head>",
            """<style>
polygon, polyline, rect, circle { fill: none !important; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
text { paint-order: stroke; stroke: #000; stroke-width: .8px; font: 16px Arial, sans-serif; }
.wrap { position: relative; display: inline-block; }
img { display: block; }
svg { position: absolute; left: 0; top: 0; }
html, body { margin: 0; padding: 0; background: #111; }
</style></head>""",
        )
    return html


def parse_svg_shapes(html: str):
    parsed = soup(html)
    svg = parsed.find("svg")
    if not svg:
        return [], (0, 0)

    def as_int(value, fallback):
        try:
            return int(float(value))
        except Exception:
            return fallback

    w = as_int(svg.get("width", "0"), 0)
    h = as_int(svg.get("height", "0"), 0)
    vb_str = svg.get("viewbox", f"0 0 {w if w > 0 else 1} {h if h > 0 else 1}")
    vb_parts = re.split(r"[ ,]+", vb_str.strip())
    vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, float(w), float(h)
    if len(vb_parts) == 4:
        try:
            vb_x, vb_y, vb_w, vb_h = [float(p) for p in vb_parts]
        except Exception:
            vb_w, vb_h = float(w), float(h)
    if vb_w <= 0:
        vb_w = float(w) if w > 0 else 1.0
    if vb_h <= 0:
        vb_h = float(h) if h > 0 else 1.0
    if w <= 0:
        w = int(vb_w)
    if h <= 0:
        h = int(vb_h)

    def transform_point(x, y):
        return ((x - vb_x) * (w / vb_w), (y - vb_y) * (h / vb_h))

    shapes: List[dict] = []
    for poly in parsed.find_all("polygon"):
        toks = poly.get("points", "").replace(",", " ").split()
        if len(toks) % 2 == 0:
            pts = [transform_point(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks), 2)]
            if pts:
                shape = {"type": "polygon", "points": pts}
                color = pick_style_color(poly)
                sw = pick_style_stroke_width(poly)
                if color:
                    shape["stroke"] = color
                if sw:
                    shape["stroke_width"] = sw
                shapes.append(shape)
    for pl in parsed.find_all("polyline"):
        toks = pl.get("points", "").replace(",", " ").split()
        if len(toks) % 2 == 0:
            pts = [transform_point(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks), 2)]
            if pts:
                shape = {"type": "polyline", "points": pts}
                color = pick_style_color(pl)
                sw = pick_style_stroke_width(pl)
                if color:
                    shape["stroke"] = color
                if sw:
                    shape["stroke_width"] = sw
                shapes.append(shape)
    for rect in parsed.find_all("rect"):
        x = float(rect.get("x", "0"))
        y = float(rect.get("y", "0"))
        rw = float(rect.get("width", "0"))
        rh = float(rect.get("height", "0"))
        x1, y1 = transform_point(x, y)
        x2, y2 = transform_point(x + rw, y + rh)
        shape = {"type": "rect", "xy": (x1, y1, x2, y2)}
        color = pick_style_color(rect)
        sw = pick_style_stroke_width(rect)
        if color:
            shape["stroke"] = color
        if sw:
            shape["stroke_width"] = sw
        shapes.append(shape)
    for circle in parsed.find_all("circle"):
        cx = float(circle.get("cx", "0"))
        cy = float(circle.get("cy", "0"))
        rr = float(circle.get("r", "0"))
        new_cx, new_cy = transform_point(cx, cy)
        avg_scale = ((w / vb_w) + (h / vb_h)) / 2.0
        shape = {"type": "circle", "center": (new_cx, new_cy), "r": rr * avg_scale}
        color = pick_style_color(circle)
        sw = pick_style_stroke_width(circle)
        if color:
            shape["stroke"] = color
        if sw:
            shape["stroke_width"] = sw
        shapes.append(shape)
    texts = []
    for text in parsed.find_all("text"):
        try:
            x, y = transform_point(float(text.get("x", "0")), float(text.get("y", "0")))
            item = {"x": x, "y": y, "text": re.sub(r"[^\x20-\x7E]", "", text.get_text(strip=True))[:24]}
            color = pick_style_color(text)
            if color:
                item["color"] = color
            texts.append(item)
        except Exception:
            pass
    if texts:
        shapes.append({"type": "text_bundle", "items": texts})
    return shapes, (w, h)


def html_from_shapes(orig_img_data_url: str, w: int, h: int, shapes: List[dict]) -> str:
    def pts_to_str(pts):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    def stroke_attr(shape):
        return f' stroke="{shape["stroke"]}"' if "stroke" in shape else ""

    def sw_attr(shape):
        return f' stroke-width="{float(shape.get("stroke_width", 3)):.2f}"' if "stroke_width" in shape else ""

    elems = []
    for shape in shapes:
        kind = shape.get("type")
        if kind == "polygon" and len(shape["points"]) >= 3:
            elems.append(f'<polygon points="{pts_to_str(shape["points"])}"{stroke_attr(shape)}{sw_attr(shape)} />')
        elif kind == "polyline" and len(shape["points"]) >= 2:
            elems.append(f'<polyline points="{pts_to_str(shape["points"])}"{stroke_attr(shape)}{sw_attr(shape)} />')
        elif kind == "rect":
            x1, y1, x2, y2 = shape["xy"]
            elems.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{(x2 - x1):.2f}" height="{(y2 - y1):.2f}"{stroke_attr(shape)}{sw_attr(shape)} />')
        elif kind == "circle":
            cx, cy = shape["center"]
            elems.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{shape["r"]:.2f}"{stroke_attr(shape)}{sw_attr(shape)} />')
        elif kind == "text_bundle":
            for item in shape["items"]:
                color = f' fill="{item["color"]}"' if "color" in item else ""
                elems.append(f'<text x="{item["x"]:.2f}" y="{item["y"]:.2f}"{color}>{item["text"]}</text>')
    svg_inner = "\n    ".join(elems)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hints</title>
<style>
html,body{{margin:0;padding:0;background:#111;}}
.wrap{{position:relative;display:inline-block}}
img{{display:block}}
svg{{position:absolute;left:0;top:0}}
polygon, polyline, rect, circle {{ fill:none; stroke-width:3; stroke-linecap:round; stroke-linejoin:round; }}
text{{ paint-order:stroke; stroke:#000; stroke-width:.8px; font:16px Arial, sans-serif }}
</style></head>
<body><div class="wrap">
  <img src="{orig_img_data_url}" width="{w}" height="{h}" />
  <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">
    {svg_inner}
  </svg>
</div></body></html>"""


def rasterize_mask(shapes: List[dict], size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        kind = shape.get("type")
        if kind == "polygon":
            pts = shape.get("points", [])
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        elif kind == "rect":
            draw.rectangle(shape.get("xy", (0, 0, 0, 0)), fill=255)
        elif kind == "circle":
            cx, cy = shape.get("center", (0, 0))
            r = shape.get("r", 0)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        elif kind == "polyline":
            pts = shape.get("points", [])
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=2)
    return np.array(mask, dtype=np.uint8)


def scale_shapes(shapes: List[dict], sx: float, sy: float) -> List[dict]:
    scale = (sx + sy) / 2.0
    out = []
    for shape in shapes or []:
        kind = shape.get("type")
        if kind in ("polygon", "polyline"):
            pts = [(float(x) * scale, float(y) * scale) for x, y in shape.get("points", [])]
            new = {"type": kind, "points": pts}
        elif kind == "rect":
            x1, y1, x2, y2 = shape.get("xy", (0, 0, 0, 0))
            new = {"type": "rect", "xy": (float(x1) * scale, float(y1) * scale, float(x2) * scale, float(y2) * scale)}
        elif kind == "circle":
            cx, cy = shape.get("center", (0, 0))
            new = {"type": "circle", "center": (float(cx) * scale, float(cy) * scale), "r": float(shape.get("r", 0.0)) * scale}
        elif kind == "text_bundle":
            items = []
            for item in shape.get("items", []):
                new_item = {"x": float(item.get("x", 0.0)) * scale, "y": float(item.get("y", 0.0)) * scale, "text": item.get("text", "")}
                if "color" in item:
                    new_item["color"] = item["color"]
                items.append(new_item)
            new = {"type": "text_bundle", "items": items}
        else:
            continue
        if "stroke" in shape:
            new["stroke"] = shape["stroke"]
        if "stroke_width" in shape:
            new["stroke_width"] = shape["stroke_width"]
        out.append(new)
    return out


def draw_shapes_on_image(base_img: Image.Image, shapes: List[dict], out_png_path: str) -> None:
    img = ensure_rgba(base_img.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    for shape in shapes:
        kind = shape.get("type")
        color = rgba_from_css(shape.get("stroke"))
        width = int(round(float(shape.get("stroke_width", 3))))
        if kind == "polygon":
            pts = shape.get("points", [])
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=color, width=width)
        elif kind == "polyline":
            pts = shape.get("points", [])
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=width)
        elif kind == "rect":
            draw.rectangle(shape.get("xy", (0, 0, 0, 0)), outline=color, width=width)
        elif kind == "circle":
            cx, cy = shape.get("center", (0, 0))
            r = shape.get("r", 0)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=width)
        elif kind == "text_bundle":
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            for item in shape.get("items", []):
                txt_color = rgba_from_css(item.get("color"))
                x, y, txt = item["x"], item["y"], item["text"] or ""
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    draw.text((x + dx, y + dy), txt, font=font, fill=(0, 0, 0, 220))
                draw.text((x, y), txt, font=font, fill=txt_color)
    img.save(out_png_path)
