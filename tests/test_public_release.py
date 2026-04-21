from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image, ImageDraw

from reason2gen_hint.cli import load_rows
from reason2gen_hint.config import RuntimeConfig
from reason2gen_hint.datasets import rows_from_parquet
from reason2gen_hint.pipeline import process_row


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)


class PublicReleaseTests(unittest.TestCase):
    def test_cli_module_help_runs(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "reason2gen_hint.cli", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Reason2Gen hint generation", proc.stdout)

    def test_rows_from_parquet_resolves_paths_relative_to_parquet_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_image(root / "images" / "orig.png", "white")
            _make_image(root / "images" / "edit.png", "black")

            df = pd.DataFrame(
                [
                    {
                        "image_id": "pq_rel",
                        "edit_instruction": "invert",
                        "image_file": "images/orig.png",
                        "edited_file": "images/edit.png",
                    }
                ]
            )
            parquet_path = root / "samples.parquet"
            df.to_parquet(parquet_path, index=False)

            rows = rows_from_parquet(str(parquet_path))
            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(rows[0].image_file["path"]).is_file())
            self.assertTrue(Path(rows[0].edited_file["path"]).is_file())

    def test_cli_load_rows_uses_image_root_for_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "parquet"
            image_root = root / "assets"
            data_dir.mkdir(parents=True, exist_ok=True)
            _make_image(image_root / "orig.png", "white")
            _make_image(image_root / "edit.png", "black")

            df = pd.DataFrame(
                [
                    {
                        "image_id": "pq_root",
                        "edit_instruction": "invert",
                        "image_file": "orig.png",
                        "edited_file": "edit.png",
                    }
                ]
            )
            df.to_parquet(data_dir / "samples.parquet", index=False)

            args = SimpleNamespace(
                dataset_source="parquet",
                data_dir=str(data_dir),
                image_root=str(image_root),
                hf_dataset=None,
                hf_split="train",
                hf_config=None,
                json_path=None,
            )
            rows = load_rows(args)
            self.assertEqual(len(rows), 1)
            self.assertEqual(Path(rows[0].image_file["path"]), image_root / "orig.png")
            self.assertEqual(Path(rows[0].edited_file["path"]), image_root / "edit.png")

    def test_multi_pass_refine_uses_diff_edges_and_overlay_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orig_path = root / "orig.png"
            edit_path = root / "edit.png"
            _make_image(orig_path, "white")
            edit_img = Image.new("RGB", (48, 48), "white")
            ImageDraw.Draw(edit_img).rectangle((8, 8, 32, 32), outline="red", width=2)
            edit_img.save(edit_path)

            row = SimpleNamespace(
                image_id="multi_case",
                image_file={"path": str(orig_path)},
                edited_file={"path": str(edit_path)},
                edit_instruction="mark the changed box",
            )
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = RuntimeConfig(mode="multi", out_root=str(out_dir))
            captured_calls = []

            def fake_build_refine_messages(
                du_o: str,
                du_e: str,
                du_diff: str,
                du_edges: str,
                du_overlay_prev: str,
                w: int,
                h: int,
                critique_text: str,
                edit_instruction: str | None,
            ):
                captured_calls.append(
                    {
                        "du_o": du_o,
                        "du_e": du_e,
                        "du_diff": du_diff,
                        "du_edges": du_edges,
                        "du_overlay_prev": du_overlay_prev,
                        "w": w,
                        "h": h,
                        "critique_text": critique_text,
                        "edit_instruction": edit_instruction,
                    }
                )
                return [{"role": "user", "content": [{"type": "text", "text": "refine"}]}]

            with (
                patch(
                    "reason2gen_hint.pipeline.gen_html_multi_pass",
                    return_value=(
                        "<html><body><svg width='48' height='48' viewBox='0 0 48 48'></svg></body></html>",
                        (48, 48),
                        "orig-url",
                        "edit-url",
                        "diff-url",
                        "edges-url",
                    ),
                ),
                patch("reason2gen_hint.pipeline.build_refine_messages", side_effect=fake_build_refine_messages),
                patch(
                    "reason2gen_hint.pipeline.gpt_chat",
                    return_value="<html><body><svg width='48' height='48' viewBox='0 0 48 48'></svg></body></html>",
                ),
                patch("reason2gen_hint.pipeline.parse_svg_shapes", return_value=([], (48, 48))),
            ):
                process_row(object(), row, cfg, str(out_dir), "multi_0")

            self.assertTrue(captured_calls, "Expected at least one refine call in multi mode")
            first = captured_calls[0]
            self.assertEqual(first["du_o"], "orig-url")
            self.assertEqual(first["du_e"], "edit-url")
            self.assertEqual(first["du_diff"], "diff-url")
            self.assertEqual(first["du_edges"], "edges-url")
            self.assertNotEqual(first["du_overlay_prev"], "orig-url")
            self.assertTrue(first["du_overlay_prev"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
