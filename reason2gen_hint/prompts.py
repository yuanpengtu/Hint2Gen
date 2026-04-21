from __future__ import annotations

from typing import List, Optional


SINGLE_PASS_SYSTEM = """You are a precision HINT sketcher for image editing.
Draw editing instructions on the ORIGINAL so that it can be transformed into the EDITED image.

Use both the edit instruction and the edited image:
- geometry and placement come from EDITED
- actions, labels, arrows, and intent come from the instruction

Output exactly one self-contained HTML file with an <img> and an absolutely positioned <svg> overlay.

Rules:
- Strokes only. Use only <polygon>, <polyline>, <rect>, <circle>, <text>.
- Stroke width 2-3px with rounded caps and joins.
- Use high-contrast colors.
- SVG size must exactly match the image size.
- Prefer tight, accurate local shapes over large coarse boxes.
- Use short English labels near edited regions.
- For remove / erase / delete instructions, explicitly outline the region to remove on ORIGINAL.
Output only the HTML."""


MULTI_PASS_SYSTEM = """You are a precision HINT sketcher for editing.
Produce one self-contained HTML overlay that shows how to transform ORIGINAL into EDITED.

Use geometry from EDITED and actions from the edit instruction.
Prefer tight local guidance, not large background boxes.
Output only the HTML."""


REFINE_SYSTEM = """You are refining a colored HINT overlay so it more precisely transforms ORIGINAL into EDITED.

Use both the edit instruction and the edited image.
Tighten geometry, remove false positives, and add missing local structures.
Output only one self-contained HTML file."""


def build_single_pass_messages(
    du_o: str,
    du_e: str,
    du_diff: str,
    du_edges: str,
    w: int,
    h: int,
    edit_instruction: Optional[str],
) -> List[dict]:
    txt = (
        "You will receive four inputs in order:\n"
        "1) ORIGINAL,\n2) EDITED,\n3) DIFF-MAP,\n4) EDGES of EDITED.\n\n"
        f"Canvas must be exactly {w}x{h}px.\n"
        "Draw hints on ORIGINAL that explain how to obtain EDITED.\n"
        "Hard constraint: polygons and polylines should cover the changed region in the DIFF-MAP.\n"
    )
    if edit_instruction:
        txt += f"\nEDIT INSTRUCTION:\n{edit_instruction.strip()}"
    return [
        {"role": "system", "content": SINGLE_PASS_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": du_o}},
                {"type": "image_url", "image_url": {"url": du_e}},
                {"type": "image_url", "image_url": {"url": du_diff}},
                {"type": "image_url", "image_url": {"url": du_edges}},
            ],
        },
    ]


def build_multi_pass_messages(
    du_o: str,
    du_e: str,
    du_diff: str,
    du_edges: str,
    w: int,
    h: int,
    edit_instruction: Optional[str],
) -> List[dict]:
    txt = (
        "You will receive ORIGINAL, EDITED, DIFF-MAP, and EDGES.\n"
        f"Canvas must be exactly {w}x{h}px.\n"
        "Draw a tight SVG hint overlay on the ORIGINAL.\n"
        "Changed geometry should agree with the DIFF-MAP.\n"
    )
    if edit_instruction:
        txt += f"\nEDIT INSTRUCTION:\n{edit_instruction.strip()}"
    return [
        {"role": "system", "content": MULTI_PASS_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": du_o}},
                {"type": "image_url", "image_url": {"url": du_e}},
                {"type": "image_url", "image_url": {"url": du_diff}},
                {"type": "image_url", "image_url": {"url": du_edges}},
            ],
        },
    ]


def build_refine_messages(
    du_o: str,
    du_e: str,
    du_diff: str,
    du_edges: str,
    du_overlay_prev: str,
    w: int,
    h: int,
    critique_text: str,
    edit_instruction: Optional[str],
) -> List[dict]:
    txt = (
        "Inputs in order: ORIGINAL, EDITED, DIFF-MAP, EDGES, PREVIOUS OVERLAY.\n"
        f"Canvas: {w}x{h}px.\n"
        f"Critique:\n{critique_text}\n"
    )
    if edit_instruction:
        txt += f"\nEDIT INSTRUCTION:\n{edit_instruction.strip()}"
    return [
        {"role": "system", "content": REFINE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": txt},
                {"type": "image_url", "image_url": {"url": du_o}},
                {"type": "image_url", "image_url": {"url": du_e}},
                {"type": "image_url", "image_url": {"url": du_diff}},
                {"type": "image_url", "image_url": {"url": du_edges}},
                {"type": "image_url", "image_url": {"url": du_overlay_prev}},
            ],
        },
    ]
