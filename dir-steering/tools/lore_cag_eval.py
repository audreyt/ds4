#!/usr/bin/env python3
"""Evaluate DS4 lore CAG across retrieval and steering parameters."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import lore_cag


DEFAULT_CASES = [
    {
        "id": "steering_build_test",
        "query": "How do I build and test a directional-steering vector?",
        "expect": ["build_direction.py", "run_sweep.py", "--good-file", "--dir-steering-file"],
        "forbid": ["/scripts/steer", "build_steering_vectors.py", "run_steering.py", "cmake"],
    },
    {
        "id": "quality_vectors",
        "query": "How are local DeepSeek V4 Flash continuation vectors compared against official vectors?",
        "expect": ["tests/test-vectors", "--dump-logprobs", "official", "top_logprobs"],
        "forbid": ["pytest", "huggingface evaluate", "wandb"],
    },
    {
        "id": "server_api",
        "query": "What API surfaces does ds4-server expose for local agents?",
        "expect": ["ds4-server", "/v1/models", "OpenAI", "Anthropic"],
        "forbid": ["FastAPI", "uvicorn", "langchain"],
    },
    {
        "id": "metal_drift",
        "query": "How should I evaluate Metal Tensor route drift and route-localize failures?",
        "expect": ["DS4_METAL_MPP_COMPARE_ROUTE", "metal-tensor-equivalence", "top-1", "RMS"],
        "forbid": ["CUDA profiler", "nsys", "torch.compile"],
    },
]


def normalize_for_match(value: str) -> str:
    value = value.lower()
    value = value.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    value = value.replace("'", "").replace("\"", "")
    value = re.sub(r"\s+", " ", value)
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_scales(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_cases(path: str) -> list[dict]:
    if not path:
        return DEFAULT_CASES
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cases"] if isinstance(payload, dict) else payload


def score_answer(answer: str, case: dict) -> dict:
    lower = normalize_for_match(answer)
    expected = case.get("expect", [])
    expect_any = case.get("expect_any", [])
    forbidden = case.get("forbid", [])
    expected_hits = [term for term in expected if normalize_for_match(term) in lower]
    any_hits: list[list[str]] = []
    any_misses: list[list[str]] = []
    for group in expect_any:
        terms = [str(term) for term in group]
        hits = [term for term in terms if normalize_for_match(term) in lower]
        if hits:
            any_hits.append(hits)
        else:
            any_misses.append(terms)
    forbidden_hits = [term for term in forbidden if normalize_for_match(term) in lower]
    citations = re.findall(r"\[\d+\]", answer)
    possible = len(expected) + len(expect_any)
    got = len(expected_hits) + len(any_hits)
    expect_score = got / max(1, possible)
    citation_bonus = min(0.15, 0.03 * len(citations))
    score = expect_score + citation_bonus - 0.25 * len(forbidden_hits)
    return {
        "score": round(score, 4),
        "expected_hits": expected_hits,
        "missing_expected": [term for term in expected if term not in expected_hits],
        "expect_any_hits": any_hits,
        "missing_expect_any": any_misses,
        "forbidden_hits": forbidden_hits,
        "citation_count": len(citations),
    }


def run_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def resolve_pack(path: str) -> Path:
    pack = Path(path)
    if not pack.is_absolute():
        pack = repo_root() / pack
    return pack


def compose_prompt(
    args: argparse.Namespace,
    pack_records: list[lore_cag.LoreRecord],
    case: dict,
    top_k: int,
    max_context_chars: int,
    prompt_path: Path,
) -> None:
    records = lore_cag.retrieve_records_from_list(
        pack_records,
        case["query"],
        top_k,
        args.min_score,
        args.after,
        args.before,
        args.neighbor_chunks,
        args.mmr_lambda,
    )
    records = lore_cag.cap_total(records, max_context_chars, args.excerpt_chars)
    prompt = lore_cag.compose_prompt(case["query"], records)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")


def predict_steering(hnet_model_dir: str, prompt_path: Path, out: Path, top_k: int, temperature: float) -> None:
    cmd = [
        sys.executable,
        str(repo_root() / "dir-steering" / "tools" / "doc_steering_hypernetwork.py"),
        "predict",
        "--model-dir", hnet_model_dir,
        "--doc", str(prompt_path),
        "--out", str(out),
        "--top-k", str(top_k),
        "--temperature", str(temperature),
    ]
    run_capture(cmd, repo_root())


def run_ds4(args: argparse.Namespace, prompt_path: Path, steering_file: Path | None, scale: str) -> tuple[str, str]:
    cmd = [
        args.ds4,
        "-m", args.model,
        "--ctx", str(args.ctx),
        "--prompt-file", str(prompt_path),
        "-n", str(args.tokens),
        "--temp", str(args.temp),
        "-sys", args.system,
        "--nothink",
    ]
    if steering_file is not None and scale != "none":
        cmd.extend([
            "--dir-steering-file", str(steering_file),
            "--dir-steering-ffn", scale,
            "--dir-steering-attn", str(args.dir_steering_attn),
        ])
    proc = run_capture(cmd, repo_root())
    return proc.stdout, proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep CAG retrieval and steering settings.")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--case-file", default="")
    ap.add_argument("--case", action="append", default=[],
                    help="case id to run; may be repeated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", default="6")
    ap.add_argument("--max-context-chars", default="9000")
    ap.add_argument("--excerpt-chars", type=int, default=1800)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--neighbor-chunks", type=int, default=0)
    ap.add_argument("--mmr-lambda", type=float, default=0.9)
    ap.add_argument("--after", default="")
    ap.add_argument("--before", default="")
    ap.add_argument("--scales", default="none")
    ap.add_argument("--hnet-model-dir", default="")
    ap.add_argument("--hnet-top-k", type=int, default=3)
    ap.add_argument("--hnet-temperature", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compose-only", action="store_true")
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=260)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--system", default="You are a precise DS4 repository assistant.")
    ap.add_argument("--dir-steering-attn", type=float, default=0.0)
    args = ap.parse_args()

    cases = load_cases(args.case_file)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        if not cases:
            raise SystemExit(f"no matching cases for {', '.join(sorted(wanted))}")
    top_ks = parse_csv_ints(args.top_k)
    contexts = parse_csv_ints(args.max_context_chars)
    scales = parse_csv_scales(args.scales)
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_records = lore_cag.load_pack(resolve_pack(args.pack))

    results: list[dict] = []
    run_count = 0
    with tempfile.TemporaryDirectory(prefix="ds4-cag-eval-") as td:
        root = Path(td)
        with out.open("w", encoding="utf-8") as f:
            for case in cases:
                for top_k in top_ks:
                    for max_context in contexts:
                        prompt_path = root / f"{case['id']}-k{top_k}-c{max_context}.txt"
                        compose_prompt(args, pack_records, case, top_k, max_context, prompt_path)
                        for scale in scales:
                            if args.limit and run_count >= args.limit:
                                break
                            run_count += 1
                            steering_file: Path | None = None
                            if args.hnet_model_dir and scale != "none":
                                steering_file = root / f"{case['id']}-k{top_k}-c{max_context}-s{scale}.f32"
                                predict_steering(args.hnet_model_dir, prompt_path, steering_file,
                                                 args.hnet_top_k, args.hnet_temperature)
                            if args.compose_only:
                                answer = prompt_path.read_text(encoding="utf-8")
                                stderr = ""
                            else:
                                answer, stderr = run_ds4(args, prompt_path, steering_file, scale)
                            scored = score_answer(answer, case)
                            result = {
                                "case": case["id"],
                                "query": case["query"],
                                "top_k": top_k,
                                "max_context_chars": max_context,
                                "scale": scale,
                                "hnet": bool(args.hnet_model_dir and scale != "none"),
                                **scored,
                                "answer": answer.strip(),
                                "stderr_tail": stderr[-1200:],
                            }
                            results.append(result)
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                        if args.limit and run_count >= args.limit:
                            break
                    if args.limit and run_count >= args.limit:
                        break
                if args.limit and run_count >= args.limit:
                    break

    print(f"wrote {out}")
    by_case: dict[str, list[dict]] = {}
    for result in results:
        by_case.setdefault(result["case"], []).append(result)
    for case_id, items in by_case.items():
        best = max(items, key=lambda item: item["score"])
        print(f"{case_id}: best score={best['score']} top_k={best['top_k']} context={best['max_context_chars']} scale={best['scale']}")


if __name__ == "__main__":
    main()
