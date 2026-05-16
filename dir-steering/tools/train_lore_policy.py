#!/usr/bin/env python3
"""Train a lightweight policy from CAG/steering outcome sweeps.

This model does not replace retrieval or steering.  It learns the small but
useful decision layer above them:

    query + candidate runtime config -> predicted answer score

At runtime, score a grid of candidate CAG/steering settings and pick the best
one.  The training labels come from lore_cag_eval.py JSONL rows, so the model is
optimized for observed DS4 outcomes rather than vector similarity.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import lore_cag


@dataclass
class Outcome:
    case: str
    query: str
    top_k: int
    max_context_chars: int
    neighbor_chunks: int
    scale: str
    hnet: bool
    score: float
    citation_count: int
    forbidden_count: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def stable_hash(text: str, feature_dim: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % feature_dim


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_scales(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_outcomes(paths: list[str]) -> list[Outcome]:
    rows: list[Outcome] = []
    for item in paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "score" not in payload or "query" not in payload:
                    raise SystemExit(f"{path}:{lineno}: missing score/query")
                rows.append(Outcome(
                    case=str(payload.get("case", f"{path.name}:{lineno}")),
                    query=str(payload["query"]),
                    top_k=int(payload.get("top_k", 0)),
                    max_context_chars=int(payload.get("max_context_chars", 0)),
                    neighbor_chunks=int(payload.get("neighbor_chunks", 0)),
                    scale=str(payload.get("scale", "none")),
                    hnet=bool(payload.get("hnet", False)),
                    score=float(payload["score"]),
                    citation_count=int(payload.get("citation_count", 0)),
                    forbidden_count=len(payload.get("forbidden_hits", [])),
                ))
    if not rows:
        raise SystemExit("no outcome rows found")
    return rows


def cjk_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    cjk = sum(1 for ch in chars if lore_cag.CJK_RE.match(ch))
    return cjk / len(chars)


def scale_bucket(scale: str) -> str:
    if scale == "none":
        return "none"
    try:
        value = float(scale)
    except ValueError:
        return scale
    if value <= -1.0:
        return "strong_neg"
    if value <= -0.5:
        return "medium_neg"
    if value < 0:
        return "light_neg"
    if value == 0:
        return "zero"
    return "positive"


def bucket_context(value: int) -> str:
    if value <= 0:
        return "unset"
    if value <= 7000:
        return "small"
    if value <= 10000:
        return "medium"
    if value <= 14000:
        return "large"
    return "xlarge"


def add_feature(counts: collections.Counter[int], name: str, feature_dim: int, value: float = 1.0) -> None:
    counts[stable_hash(name, feature_dim)] += value


def features(row: Outcome, feature_dim: int) -> dict[int, float]:
    counts: collections.Counter[int] = collections.Counter()
    add_feature(counts, "bias", feature_dim)

    scale = scale_bucket(row.scale)
    context = bucket_context(row.max_context_chars)
    lang = "cjk" if cjk_ratio(row.query) >= 0.2 else "latin"

    config_names = [
        f"top_k={row.top_k}",
        f"context={context}",
        f"neighbor={row.neighbor_chunks}",
        f"scale={scale}",
        f"hnet={int(row.hnet)}",
        f"lang={lang}",
        f"top_k={row.top_k}|context={context}",
        f"scale={scale}|hnet={int(row.hnet)}",
        f"scale={scale}|lang={lang}",
    ]
    for name in config_names:
        add_feature(counts, f"cfg:{name}", feature_dim)

    tokens = lore_cag.tokenize(row.query)
    token_counts = collections.Counter(tokens)
    for token, count in token_counts.items():
        weight = 1.0 + math.log(count)
        add_feature(counts, f"q:{token}", feature_dim, weight)
        add_feature(counts, f"q:{token}|scale={scale}", feature_dim, 0.35 * weight)
        add_feature(counts, f"q:{token}|context={context}", feature_dim, 0.25 * weight)

    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm <= 0:
        return {}
    return {idx: value / norm for idx, value in counts.items()}


def dot(weights: dict[int, float], x: dict[int, float]) -> float:
    if len(weights) < len(x):
        return sum(value * x.get(idx, 0.0) for idx, value in weights.items())
    return sum(value * weights.get(idx, 0.0) for idx, value in x.items())


def train_model(rows: list[Outcome], feature_dim: int, epochs: int, lr: float, l2: float, seed: int) -> tuple[dict[int, float], list[float]]:
    rng = random.Random(seed)
    examples = [(features(row, feature_dim), row.score) for row in rows]
    weights: dict[int, float] = {}
    losses: list[float] = []
    order = list(range(len(examples)))
    for _ in range(max(1, epochs)):
        rng.shuffle(order)
        loss = 0.0
        for i in order:
            x, target = examples[i]
            pred = dot(weights, x)
            err = pred - target
            loss += err * err
            for idx, value in x.items():
                grad = err * value + l2 * weights.get(idx, 0.0)
                weights[idx] = weights.get(idx, 0.0) - lr * grad
        losses.append(loss / max(1, len(examples)))
    return weights, losses


def predict_score(weights: dict[int, float], row: Outcome, feature_dim: int) -> float:
    return dot(weights, features(row, feature_dim))


def policy_metrics(rows: list[Outcome], weights: dict[int, float], feature_dim: int) -> dict:
    by_case: dict[str, list[Outcome]] = collections.defaultdict(list)
    for row in rows:
        by_case[row.case].append(row)

    regrets: list[float] = []
    chosen_scores: list[float] = []
    oracle_scores: list[float] = []
    exact = 0
    for items in by_case.values():
        oracle = max(items, key=lambda row: row.score)
        chosen = max(items, key=lambda row: predict_score(weights, row, feature_dim))
        regrets.append(oracle.score - chosen.score)
        chosen_scores.append(chosen.score)
        oracle_scores.append(oracle.score)
        if chosen.score == oracle.score:
            exact += 1
    return {
        "case_count": len(by_case),
        "mean_chosen_score": round(statistics.mean(chosen_scores), 4) if chosen_scores else 0.0,
        "mean_oracle_score": round(statistics.mean(oracle_scores), 4) if oracle_scores else 0.0,
        "mean_regret": round(statistics.mean(regrets), 4) if regrets else 0.0,
        "max_regret": round(max(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact / max(1, len(by_case)), 4),
    }


def leave_one_case_out(rows: list[Outcome], feature_dim: int, epochs: int, lr: float, l2: float, seed: int) -> dict:
    cases = sorted({row.case for row in rows})
    regrets: list[float] = []
    exact = 0
    skipped = 0
    for case in cases:
        train_rows = [row for row in rows if row.case != case]
        heldout = [row for row in rows if row.case == case]
        if not train_rows or len(heldout) < 2:
            skipped += 1
            continue
        weights, _ = train_model(train_rows, feature_dim, epochs, lr, l2, seed)
        oracle = max(heldout, key=lambda row: row.score)
        chosen = max(heldout, key=lambda row: predict_score(weights, row, feature_dim))
        regrets.append(oracle.score - chosen.score)
        if chosen.score == oracle.score:
            exact += 1
    return {
        "case_count": len(cases) - skipped,
        "skipped": skipped,
        "mean_regret": round(statistics.mean(regrets), 4) if regrets else 0.0,
        "max_regret": round(max(regrets), 4) if regrets else 0.0,
        "exact_best_rate": round(exact / max(1, len(regrets)), 4),
    }


def save_model(model_dir: Path, payload: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(model_dir: Path) -> dict:
    model_path = model_dir / "model.json"
    if not model_path.exists():
        raise SystemExit(f"{model_path}: missing model")
    return json.loads(model_path.read_text(encoding="utf-8"))


def outcome_grid(query: str, top_ks: list[int], contexts: list[int], neighbor_chunks: list[int], scales: list[str]) -> list[Outcome]:
    rows: list[Outcome] = []
    for top_k in top_ks:
        for context in contexts:
            for neighbor in neighbor_chunks:
                for scale in scales:
                    rows.append(Outcome(
                        case="predict",
                        query=query,
                        top_k=top_k,
                        max_context_chars=context,
                        neighbor_chunks=neighbor,
                        scale=scale,
                        hnet=scale != "none",
                        score=0.0,
                        citation_count=0,
                        forbidden_count=0,
                    ))
    return rows


def train_cmd(args: argparse.Namespace) -> None:
    rows = load_outcomes(args.outcome)
    weights, losses = train_model(rows, args.feature_dim, args.epochs, args.lr, args.l2, args.seed)
    metrics = policy_metrics(rows, weights, args.feature_dim)
    loo = leave_one_case_out(rows, args.feature_dim, args.loo_epochs, args.lr, args.l2, args.seed) if args.leave_one_case_out else None

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = repo_root() / model_dir
    payload = {
        "format": "ds4-lore-outcome-policy-v1",
        "feature_dim": args.feature_dim,
        "weights": [[idx, value] for idx, value in sorted(weights.items()) if abs(value) > 1e-12],
        "training": {
            "rows": len(rows),
            "cases": len({row.case for row in rows}),
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "metrics": metrics,
            "leave_one_case_out": loo,
        },
        "default_grid": {
            "top_k": parse_csv_ints(args.default_top_k),
            "max_context_chars": parse_csv_ints(args.default_max_context_chars),
            "neighbor_chunks": parse_csv_ints(args.default_neighbor_chunks),
            "scales": parse_csv_scales(args.default_scales),
        },
    }
    save_model(model_dir, payload)
    print(f"wrote {model_dir / 'model.json'}")
    print(f"trained on {len(rows)} rows / {payload['training']['cases']} cases loss {losses[0]:0.4f}->{losses[-1]:0.4f}")
    print(f"train policy: chosen={metrics['mean_chosen_score']} oracle={metrics['mean_oracle_score']} regret={metrics['mean_regret']} exact={metrics['exact_best_rate']}")
    if loo:
        print(f"leave-one-case-out: regret={loo['mean_regret']} max={loo['max_regret']} exact={loo['exact_best_rate']} cases={loo['case_count']}")


def eval_cmd(args: argparse.Namespace) -> None:
    rows = load_outcomes(args.outcome)
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = repo_root() / model_dir
    model = load_model(model_dir)
    feature_dim = int(model["feature_dim"])
    weights = {int(idx): float(value) for idx, value in model.get("weights", [])}
    metrics = policy_metrics(rows, weights, feature_dim)
    print(json.dumps(metrics, indent=2))


def predict_cmd(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = repo_root() / model_dir
    model = load_model(model_dir)
    feature_dim = int(model["feature_dim"])
    weights = {int(idx): float(value) for idx, value in model.get("weights", [])}
    defaults = model.get("default_grid", {})
    top_ks = parse_csv_ints(args.top_k) if args.top_k else [int(x) for x in defaults.get("top_k", [3, 5])]
    contexts = parse_csv_ints(args.max_context_chars) if args.max_context_chars else [int(x) for x in defaults.get("max_context_chars", [8000, 12000])]
    neighbors = parse_csv_ints(args.neighbor_chunks) if args.neighbor_chunks else [int(x) for x in defaults.get("neighbor_chunks", [1])]
    scales = parse_csv_scales(args.scales) if args.scales else [str(x) for x in defaults.get("scales", ["none", "-0.75"])]
    candidates = outcome_grid(args.query, top_ks, contexts, neighbors, scales)
    scored = sorted(
        [
            {
                "predicted_score": round(predict_score(weights, row, feature_dim), 4),
                "top_k": row.top_k,
                "max_context_chars": row.max_context_chars,
                "neighbor_chunks": row.neighbor_chunks,
                "scale": row.scale,
                "hnet": row.hnet,
            }
            for row in candidates
        ],
        key=lambda item: item["predicted_score"],
        reverse=True,
    )
    payload = {
        "query": args.query,
        "best": scored[0],
        "candidates": scored[:args.show],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = repo_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train/query a CAG steering outcome policy.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    train_ap = sub.add_parser("train")
    train_ap.add_argument("--outcome", action="append", required=True)
    train_ap.add_argument("--model-dir", required=True)
    train_ap.add_argument("--feature-dim", type=int, default=4096)
    train_ap.add_argument("--epochs", type=int, default=160)
    train_ap.add_argument("--loo-epochs", type=int, default=80)
    train_ap.add_argument("--lr", type=float, default=0.08)
    train_ap.add_argument("--l2", type=float, default=0.0001)
    train_ap.add_argument("--seed", type=int, default=7)
    train_ap.add_argument("--leave-one-case-out", action="store_true")
    train_ap.add_argument("--default-top-k", default="3,5")
    train_ap.add_argument("--default-max-context-chars", default="8000,12000")
    train_ap.add_argument("--default-neighbor-chunks", default="1")
    train_ap.add_argument("--default-scales", default="none,-0.25,-0.75,-1.0")
    train_ap.set_defaults(func=train_cmd)

    eval_ap = sub.add_parser("eval")
    eval_ap.add_argument("--model-dir", required=True)
    eval_ap.add_argument("--outcome", action="append", required=True)
    eval_ap.set_defaults(func=eval_cmd)

    predict_ap = sub.add_parser("predict")
    predict_ap.add_argument("--model-dir", required=True)
    predict_ap.add_argument("--query", required=True)
    predict_ap.add_argument("--top-k", default="")
    predict_ap.add_argument("--max-context-chars", default="")
    predict_ap.add_argument("--neighbor-chunks", default="")
    predict_ap.add_argument("--scales", default="")
    predict_ap.add_argument("--show", type=int, default=6)
    predict_ap.add_argument("--out", default="")
    predict_ap.set_defaults(func=predict_cmd)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
