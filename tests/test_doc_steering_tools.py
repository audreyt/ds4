#!/usr/bin/env python3
"""Reliability tests for the local doc-to-steering tooling."""

from __future__ import annotations

import array
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_BUILDER = ROOT / "dir-steering" / "tools" / "build_doc_direction.py"
HYPERNET = ROOT / "dir-steering" / "tools" / "doc_steering_hypernetwork.py"
LORE_CAG = ROOT / "dir-steering" / "tools" / "lore_cag.py"
LORE_EVAL = ROOT / "dir-steering" / "tools" / "lore_cag_eval.py"
LORE_TESTSET = ROOT / "dir-steering" / "tools" / "build_lore_testset.py"
N_LAYER = 43
N_EMBD = 4096
VECTOR_FLOATS = N_LAYER * N_EMBD


def write_onehot_vector(path: Path, column: int) -> None:
    data = array.array("f", [0.0]) * VECTOR_FLOATS
    for layer in range(N_LAYER):
        data[layer * N_EMBD + column] = 1.0
    with path.open("wb") as f:
        data.tofile(f)


def read_vector(path: Path) -> array.array:
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, VECTOR_FLOATS)
    return data


def average_layer_cosine(a: array.array, b: array.array) -> float:
    total = 0.0
    for layer in range(N_LAYER):
        start = layer * N_EMBD
        end = start + N_EMBD
        dot = 0.0
        an = 0.0
        bn = 0.0
        for av, bv in zip(a[start:end], b[start:end]):
            dot += float(av) * float(bv)
            an += float(av) * float(av)
            bn += float(bv) * float(bv)
        total += dot / math.sqrt(an * bn)
    return total / N_LAYER


class DocSteeringToolTests(unittest.TestCase):
    def test_doc_builder_writes_balanced_prompt_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.md"
            b = root / "b.md"
            a.write_text("# Alpha\n\nDS4 alpha_route " * 80, encoding="utf-8")
            b.write_text("# Beta\n\nDS4 beta_route " * 80, encoding="utf-8")
            out = root / "built.json"

            subprocess.run([
                sys.executable,
                str(DOC_BUILDER),
                "--doc", str(a),
                "--doc", str(b),
                "--out", str(out),
                "--max-prompts", "2",
                "--max-prompts-per-file", "1",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            good = out.with_name("built.doc-good.txt")
            bad = out.with_name("built.doc-bad.txt")
            recipe = out.with_name("built.doc-to-steering.json")
            self.assertEqual(len(good.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(bad.read_text(encoding="utf-8").splitlines()), 2)
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            self.assertEqual(payload["prompt_count"], 2)
            got_paths = {str(Path(chunk["path"]).resolve()) for chunk in payload["chunks"]}
            self.assertEqual(got_paths, {str(a.resolve()), str(b.resolve())})

    def test_hypernetwork_predicts_normalized_matching_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            recipes = []
            for name, token, column in [
                ("alpha", "alpha_route alpha_route maintainer", 7),
                ("beta", "beta_cache beta_cache generic", 23),
            ]:
                good = root / f"{name}.good.txt"
                bad = root / f"{name}.bad.txt"
                out = root / f"{name}.json"
                vec = root / f"{name}.f32"
                recipe = root / f"{name}.doc-to-steering.json"
                good.write_text(token, encoding="utf-8")
                bad.write_text("ordinary software explanation", encoding="utf-8")
                write_onehot_vector(vec, column)
                recipe.write_text(json.dumps({
                    "format": "ds4-doc-to-steering-prompts-v1",
                    "good_file": str(good),
                    "bad_file": str(bad),
                    "out": str(out),
                    "chunks": [{"anchor": token, "terms": token.split()}],
                    "docs": [],
                }), encoding="utf-8")
                recipes.append(recipe)
            (root / "prediction.json").write_text(json.dumps({
                "format": "not-a-training-recipe",
                "out": str(root / "prediction.f32"),
            }), encoding="utf-8")

            model_dir = root / "model"
            subprocess.run([
                sys.executable,
                str(HYPERNET),
                "train",
                "--recipe", str(root),
                "--model-dir", str(model_dir),
                "--feature-dim", "256",
                "--top-k", "1",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            pred = root / "pred.f32"
            subprocess.run([
                sys.executable,
                str(HYPERNET),
                "predict",
                "--model-dir", str(model_dir),
                "--doc", "alpha_route maintainer alpha_route",
                "--out", str(pred),
                "--top-k", "1",
                "--temperature", "0",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            predicted = read_vector(pred)
            gold = read_vector(root / "alpha.f32")
            self.assertEqual(len(predicted), VECTOR_FLOATS)
            self.assertGreater(average_layer_cosine(predicted, gold), 0.99)

            for layer in range(N_LAYER):
                start = layer * N_EMBD
                end = start + N_EMBD
                norm = math.sqrt(sum(float(x) * float(x) for x in predicted[start:end]))
                self.assertAlmostEqual(norm, 1.0, places=5)

    def test_lore_cag_retrieves_and_composes_cited_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alpha = root / "alpha.md"
            beta = root / "beta.md"
            alpha.write_text(
                "# Directional Steering\n\n"
                "Use dir-steering/tools/build_direction.py with --good-file and --bad-file "
                "to capture DS4 ffn_out directions. The sweep helper is run_sweep.py.\n",
                encoding="utf-8",
            )
            beta.write_text(
                "# Server Lore\n\n"
                "Use ds4-server for OpenAI-compatible HTTP serving and tool-call mapping.\n",
                encoding="utf-8",
            )
            pack = root / "pack.jsonl"

            subprocess.run([
                sys.executable,
                str(LORE_CAG),
                "pack",
                "--doc", str(alpha),
                "--doc", str(beta),
                "--out", str(pack),
                "--max-chunk-chars", "500",
                "--min-chunk-chars", "20",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            retrieved = root / "retrieved.json"
            subprocess.run([
                sys.executable,
                str(LORE_CAG),
                "retrieve",
                "--pack", str(pack),
                "--query", "How do I build a directional steering vector with good-file bad-file?",
                "--top-k", "1",
                "--out", str(retrieved),
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            payload = json.loads(retrieved.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 1)
            self.assertTrue(payload["records"][0]["path"].endswith("alpha.md"))
            self.assertIn("build_direction.py", payload["records"][0]["text"])

            prompt = root / "prompt.txt"
            subprocess.run([
                sys.executable,
                str(LORE_CAG),
                "prompt",
                "--pack", str(pack),
                "--query", "What script builds a directional steering vector?",
                "--top-k", "1",
                "--out", str(prompt),
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            body = prompt.read_text(encoding="utf-8")
            self.assertIn("Use only the cited lore excerpts", body)
            self.assertIn("build_direction.py", body)
            self.assertIn("[1] file:", body)

    def test_lore_cag_supports_date_filters_cjk_and_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "2024-05-01-old.md"
            alpha = root / "2024-06-13-alpha.md"
            old.write_text("# Old\n\nThis old note mentions 演算法 but should be outside the range.\n", encoding="utf-8")
            alpha.write_text(
                "# 讓演算法變得善良\n\n"
                "第一段只介紹背景。\n\n"
                "唐鳳說，公民 AI 可以讓演算法變得善良，透過公共監督和社群評估降低極化。\n",
                encoding="utf-8",
            )
            pack = root / "pack.jsonl"
            retrieved = root / "retrieved.json"

            subprocess.run([
                sys.executable, str(LORE_CAG), "pack",
                "--doc", str(root), "--out", str(pack),
                "--after", "2024-05-20",
                "--max-chunk-chars", "80", "--min-chunk-chars", "8",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run([
                sys.executable, str(LORE_CAG), "retrieve",
                "--pack", str(pack),
                "--query", "哪一篇談到讓演算法變得善良與公民 AI？",
                "--top-k", "1",
                "--neighbor-chunks", "1",
                "--out", str(retrieved),
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            payload = json.loads(retrieved.read_text(encoding="utf-8"))
            self.assertTrue(payload["records"])
            self.assertTrue(all("2024-05-01-old.md" not in item["path"] for item in payload["records"]))
            self.assertEqual(payload["records"][0]["source_date"], "2024-06-13")
            self.assertIn("演算法", "\n".join(item["text"] for item in payload["records"]))

    def test_lore_testset_builds_honest_compose_only_cases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "2025-01-03-civic-ai.md"
            second = root / "2026-05-08-local-kami.md"
            first.write_text(
                "# Civic AI Assembly\n\n"
                "The assembly described a quartz mailbox protocol for civic AI evaluations, "
                "with public logs and community-authored release gates.\n",
                encoding="utf-8",
            )
            second.write_text(
                "# Local Kami Systems\n\n"
                "The interview described bounded local Kami systems with charters, resource limits, "
                "appeal paths, and sunset conditions for communities.\n",
                encoding="utf-8",
            )
            case_file = root / "cases.json"
            pack = root / "pack.jsonl"
            out = root / "eval.jsonl"

            subprocess.run([
                sys.executable, str(LORE_TESTSET),
                "--doc", str(root), "--out", str(case_file),
                "--after", "2025-01-01", "--before", "2026-05-16",
                "--limit", "2",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cases_payload = json.loads(case_file.read_text(encoding="utf-8"))
            self.assertEqual(cases_payload["case_count"], 2)
            for case in cases_payload["cases"]:
                self.assertNotIn(case["source_date"], case["query"])
                self.assertIn("expect_any", case)

            subprocess.run([
                sys.executable, str(LORE_CAG), "pack",
                "--doc", str(root), "--out", str(pack),
                "--after", "2025-01-01", "--before", "2026-05-16",
                "--max-chunk-chars", "500", "--min-chunk-chars", "20",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run([
                sys.executable, str(LORE_EVAL),
                "--pack", str(pack),
                "--case-file", str(case_file),
                "--out", str(out),
                "--top-k", "2",
                "--compose-only",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["score"] >= 1.0 for row in rows))

    def test_lore_eval_compose_only_scores_expected_terms(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            alpha = root / "alpha.md"
            alpha.write_text(
                "# Directional Steering\n\n"
                "Use dir-steering/tools/build_direction.py and dir-steering/tools/run_sweep.py "
                "with --good-file and --dir-steering-file.\n",
                encoding="utf-8",
            )
            pack = root / "pack.jsonl"
            cases = root / "cases.json"
            out = root / "eval.jsonl"

            subprocess.run([
                sys.executable, str(LORE_CAG), "pack",
                "--doc", str(alpha), "--out", str(pack),
                "--max-chunk-chars", "500", "--min-chunk-chars", "20",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cases.write_text(json.dumps({"cases": [{
                "id": "steer",
                "query": "How do I build and test a directional-steering vector?",
                "expect": ["build_direction.py", "run_sweep.py", "--good-file", "--dir-steering-file"],
                "forbid": ["build_steering_vectors.py"],
            }]}), encoding="utf-8")
            subprocess.run([
                sys.executable, str(LORE_EVAL),
                "--pack", str(pack),
                "--case-file", str(cases),
                "--out", str(out),
                "--compose-only",
            ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(rows[0]["score"], 1.0)
            self.assertFalse(rows[0]["forbidden_hits"])


if __name__ == "__main__":
    unittest.main()
