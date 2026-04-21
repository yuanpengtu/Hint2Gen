from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from .config import (
    GRID_CELL_DEFAULT,
    GRID_CELL_MAX,
    GRID_CELL_MIN,
    MAX_DATAURL_CHARS,
    ROI_CELL_MAX,
    ROI_CELL_MIN,
    ROI_PAD_FRAC,
)
from .io_utils import encode_jpeg, resize, to_data_url


def register_images(o_img: Image.Image, e_img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    w, h = size
    o_rgb = np.array(o_img.convert("RGB").resize((w, h), Image.LANCZOS))
    e_rgb = np.array(e_img.convert("RGB").resize((w, h), Image.LANCZOS))
    o_g = cv2.cvtColor(o_rgb, cv2.COLOR_RGB2GRAY)
    e_g = cv2.cvtColor(e_rgb, cv2.COLOR_RGB2GRAY)

    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(e_g, None)
        kp2, des2 = sift.detectAndCompute(o_g, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            raise cv2.error("SIFT failed to find enough descriptors.")
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) <= 10:
            raise cv2.error(f"Not enough good matches found with SIFT - {len(good)}/10")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if matrix is None:
            raise cv2.error("findHomography failed after SIFT matching.")
        return Image.fromarray(cv2.warpPerspective(e_rgb, matrix, (w, h)))
    except Exception:
        try:
            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(e_g, None)
            kp2, des2 = orb.detectAndCompute(o_g, None)
            if des1 is None or des2 is None:
                raise cv2.error("ORB found no descriptors.")
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
            if len(matches) < 10:
                raise cv2.error("Not enough ORB matches.")
            good = matches[:50]
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if matrix is None:
                raise cv2.error("findHomography failed after ORB.")
            return Image.fromarray(cv2.warpPerspective(e_rgb, matrix, (w, h)))
        except Exception:
            return Image.fromarray(e_rgb)


def ssim_map_gray(a_g_u8: np.ndarray, b_g_u8: np.ndarray) -> np.ndarray:
    a = a_g_u8.astype(np.float32)
    b = b_g_u8.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = 11
    sigma = 1.5
    mu1 = cv2.GaussianBlur(a, (win, win), sigma)
    mu2 = cv2.GaussianBlur(b, (win, win), sigma)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(a * a, (win, win), sigma) - mu1_2
    sigma2_2 = cv2.GaussianBlur(b * b, (win, win), sigma) - mu2_2
    sigma12 = cv2.GaussianBlur(a * b, (win, win), sigma) - mu12
    num = (2 * mu12 + c1) * (2 * sigma12 + c2)
    den = (mu1_2 + mu2_2 + c1) * (sigma1_2 + sigma2_2 + c2) + 1e-9
    return np.clip(num / den, 0.0, 1.0)


def make_diff_precise(
    o_img: Image.Image, e_img: Image.Image, size: Tuple[int, int]
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    w, h = size
    e_aligned_img = register_images(o_img, e_img, (w, h))
    o_rgb = np.array(o_img.convert("RGB").resize((w, h), Image.LANCZOS))
    e_rgb = np.array(e_aligned_img)

    o_lab = cv2.cvtColor(o_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    e_lab = cv2.cvtColor(e_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d_e = np.clip(np.linalg.norm(o_lab - e_lab, axis=2) / 100.0, 0.0, 1.0)

    o_g = cv2.cvtColor(o_rgb, cv2.COLOR_RGB2GRAY)
    e_g = cv2.cvtColor(e_rgb, cv2.COLOR_RGB2GRAY)
    d_struct = 1.0 - ssim_map_gray(o_g, e_g)
    edge_o = cv2.Canny(o_g, 60, 140).astype(np.float32) / 255.0
    edge_e = cv2.Canny(e_g, 60, 140).astype(np.float32) / 255.0
    edge_delta = np.clip(edge_o + edge_e - 2.0 * (edge_o * edge_e), 0.0, 1.0)

    score = 0.55 * d_e + 0.35 * d_struct + 0.10 * edge_delta
    score_u8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)

    bw = cv2.adaptiveThreshold(score_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2)
    k3 = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(32, int(0.0004 * w * h))
    mask = np.zeros((h, w), dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    dist_to_edge = cv2.distanceTransform(255 - (edge_e * 255).astype(np.uint8), cv2.DIST_L2, 3)
    erode = cv2.erode(mask, k3, iterations=1)
    boundary = cv2.absdiff(mask, erode)
    snap_zone = (dist_to_edge < 2.5).astype(np.uint8) * 255
    mask[boundary > 0] = np.maximum(mask[boundary > 0], snap_zone[boundary > 0])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

    edge_outline = cv2.Canny(mask, 0, 1)
    overlay = o_rgb.copy()
    overlay[edge_outline > 0] = (0, 229, 255)
    return mask, Image.fromarray(overlay), Image.fromarray(e_rgb)


def diff_mask(a_img: Image.Image, b_img: Image.Image) -> np.ndarray:
    size = (b_img.width, b_img.height)
    mask, _, _ = make_diff_precise(a_img, b_img, size)
    return mask


def auto_contrast(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.autocontrast(img, cutoff=2)
    except Exception:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
        if p98 <= p2:
            return img
        scale = 255.0 / float(p98 - p2)
        lut = []
        for i in range(256):
            v = int((i - p2) * scale)
            lut.append(0 if v < 0 else (255 if v > 255 else v))
        return img.convert("L").point(lut).convert("RGB")


def make_edges(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    im = img.convert("RGB").resize(size, Image.LANCZOS)
    return auto_contrast(im.filter(ImageFilter.FIND_EDGES))


def add_grid(img: Image.Image, cell: int = GRID_CELL_DEFAULT) -> Image.Image:
    g = img.convert("RGBA").copy()
    w, h = g.size
    draw = ImageDraw.Draw(g, "RGBA")
    line_color = (255, 255, 255, 90)
    axis_color = (255, 77, 79, 140)
    for x in range(0, w, int(cell)):
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
    for y in range(0, h, int(cell)):
        draw.line([(0, y), (w, y)], fill=line_color, width=1)
    draw.line([(0, 0), (w, 0)], fill=axis_color, width=2)
    draw.line([(0, 0), (0, h)], fill=axis_color, width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    step = int(cell) * 3 if cell > 0 else 192
    for x in range(0, w, step):
        label = f"x={x}"
        draw.text((x + 3, 3), label, fill=(0, 0, 0, 180), font=font)
        draw.text((x + 2, 2), label, fill=(255, 255, 255, 230), font=font)
    for y in range(0, h, step):
        label = f"y={y}"
        draw.text((3, y + 3), label, fill=(0, 0, 0, 180), font=font)
        draw.text((2, y + 2), label, fill=(255, 255, 255, 230), font=font)
    return g


def compute_roi(o_img: Image.Image, e_img: Image.Image, size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    o = o_img.convert("RGB").resize(size, Image.LANCZOS)
    e = e_img.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(ImageChops.difference(o, e).convert("L"), dtype=np.uint8)
    thr = max(18, int(arr.mean() + arr.std() * 0.70))
    ys, xs = np.nonzero((arr > thr).astype(np.uint8))
    if xs.size == 0:
        return None
    w, h = size
    pad = int(max(w, h) * ROI_PAD_FRAC)
    return (
        max(0, xs.min() - pad),
        max(0, ys.min() - pad),
        min(w - 1, xs.max() + pad),
        min(h - 1, ys.max() + pad),
    )


def add_grid_adaptive(
    img: Image.Image,
    base_cell: Optional[int] = None,
    roi_box: Optional[Tuple[int, int, int, int]] = None,
    mode: str = "adaptive",
) -> Image.Image:
    if mode == "off":
        return img
    g = img.convert("RGBA").copy()
    w, h = g.size
    draw = ImageDraw.Draw(g, "RGBA")
    cell_g = int(
        np.clip(round(max(w, h) / 20.0), GRID_CELL_MIN, GRID_CELL_MAX)
        if not base_cell
        else np.clip(base_cell, GRID_CELL_MIN, GRID_CELL_MAX)
    )
    for x in range(0, w, cell_g):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 72), width=1)
    for y in range(0, h, cell_g):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 72), width=1)
    draw.line([(0, 0), (w, 0)], fill=(255, 77, 79, 150), width=2)
    draw.line([(0, 0), (0, h)], fill=(255, 77, 79, 150), width=2)
    try:
        font_big = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    for x in range(0, w, cell_g * 3):
        label = f"x={x}"
        draw.text((x + 2, 2), label, fill=(255, 255, 255, 230), font=font_big)
    for y in range(0, h, cell_g * 3):
        label = f"y={y}"
        draw.text((2, y + 2), label, fill=(255, 255, 255, 230), font=font_big)
    if mode != "adaptive" or roi_box is None:
        return g
    x1, y1, x2, y2 = roi_box
    rw, rh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    cell_r = int(np.clip(cell_g // 2, ROI_CELL_MIN, ROI_CELL_MAX))
    if (rw * rh) / float(w * h) > 0.35:
        cell_r = cell_g
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255, 140), width=2)
    gx0 = x1 - (x1 % cell_r)
    gy0 = y1 - (y1 % cell_r)
    for x in range(gx0, x2 + 1, cell_r):
        draw.line([(x, y1), (x, y2)], fill=(0, 255, 255, 80), width=1)
    for y in range(gy0, y2 + 1, cell_r):
        draw.line([(x1, y), (x2, y)], fill=(0, 255, 255, 80), width=1)
    for x in range(gx0, x2, cell_r):
        for y in range(gy0, y2, cell_r):
            mx, my = x + cell_r // 2, y + cell_r // 2
            if x1 <= mx <= x2 and y1 <= my <= y2:
                draw.text((mx - 18, my - 8), f"{mx},{my}", fill=(255, 255, 255, 230), font=font_small)
    return g


def make_adaptive_payload(
    o_img: Image.Image,
    e_img: Image.Image,
    max_side_start: int = 1600,
    with_grid: bool = True,
    grid_mode: str = "on",
    grid_cell: int = GRID_CELL_DEFAULT,
):
    sides = [max_side_start, 1536, 1400, 1280, 1152, 1024, 960, 896, 832, 768, 640, 512, 384]
    qualities = [90, 85, 80, 72, 65]

    def apply_grid(img: Image.Image) -> Image.Image:
        if not with_grid:
            return img
        gm = (grid_mode or "on").lower()
        if gm == "off":
            return img
        if gm == "auto":
            roi = compute_roi(o_img, e_img, img.size)
            return add_grid_adaptive(img, base_cell=grid_cell, roi_box=roi, mode="adaptive")
        return add_grid(img, cell=int(grid_cell))

    for side in sides:
        o_s = apply_grid(resize(o_img, side))
        e_s = apply_grid(resize(e_img, side))
        for quality in qualities:
            du_o = to_data_url(encode_jpeg(o_s, quality), "image/jpeg")
            du_e = to_data_url(encode_jpeg(e_s, quality), "image/jpeg")
            if len(du_o) + len(du_e) <= MAX_DATAURL_CHARS:
                return du_o, du_e, o_s.size
    o_s = apply_grid(resize(o_img, 384))
    e_s = apply_grid(resize(e_img, 384))
    return to_data_url(encode_jpeg(o_s, 65)), to_data_url(encode_jpeg(e_s, 65)), o_s.size
