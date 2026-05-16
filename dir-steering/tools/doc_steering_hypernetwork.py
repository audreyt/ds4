#!/usr/bin/env python3
"""Tiny dependency-free hypernetwork for document-conditioned DS4 steering.

The model is intentionally simple and local:

    document text -> hashed sparse features -> example coefficients
      -> weighted steering-vector basis -> normalized 43 x 4096 direction

Training consumes examples produced by build_doc_direction.py or
build_direction.py.  Each example contributes text features and one gold
steering vector.  Prediction mixes the closest gold vectors and normalizes each
layer.  This is a practical first hypernetwork: cheap, inspectable, and useful
as a reliability baseline before trying a neural encoder.
"""

from __future__ import annotations

import argparse
import array
import collections
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


N_LAYER = 43
N_EMBD = 4096
VECTOR_FLOATS = N_LAYER * N_EMBD
VECTOR_BYTES = VECTOR_FLOATS * 4
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}|[0-9]{2,}")

STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "before", "being", "but", "can", "default", "does", "each", "else",
    "false", "file", "for", "from", "have", "help", "here", "into", "long",
    "make", "may", "more", "must", "not", "one", "only", "other", "path",
    "read", "return", "same", "should", "that", "the", "then", "there",
    "this", "true", "use", "used", "using", "when", "where", "with", "would",
}


@dataclass
class Example:
    name: str
    recipe_path: str
    vector_file: str
    features: dict[int, float]
    text_chars: int


def stable_hash(text: str, feature_dim: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % feature_dim


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if len(part) > 2 and part not in STOPWORDS)
    return tokens


def raw_feature_counts(text: str, feature_dim: int) -> collections.Counter[int]:
    counts: collections.Counter[int] = collections.Counter()
    for token in tokenize(text):
        counts[stable_hash(token, feature_dim)] += 1
    return counts


def normalize_sparse(counts: dict[int, float], idf: dict[int, float] | None = None) -> dict[int, float]:
    weighted: dict[int, float] = {}
    for idx, value in counts.items():
        if value <= 0:
            continue
        tf = 1.0 + math.log(float(value))
        weighted[idx] = tf * (idf.get(idx, 1.0) if idf else 1.0)
    norm = math.sqrt(sum(v * v for v in weighted.values()))
    if norm <= 0:
        return {}
    return {idx: value / norm for idx, value in weighted.items()}


def sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(idx, 0.0) for idx, value in a.items())


def read_text_file(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except UnicodeDecodeError:
        text = path.read_bytes().decode("utf-8", errors="ignore")
    if max_chars > 0 and len(text) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + "\n\n[... omitted ...]\n\n" + text[-tail:]
    return text


def load_recipe(path: Path, max_chars: int) -> tuple[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []

    for key in ("good_file", "bad_file"):
        value = payload.get(key)
        if value:
            p = Path(value)
            if not p.is_absolute():
                p = path.parent / p
            if p.exists():
                # Target probes carry the corpus-specific register; contrast
                # probes help the encoder see what should be separated away.
                parts.append(read_text_file(p, max_chars))

    for chunk in payload.get("chunks", []):
        parts.append(str(chunk.get("anchor", "")))
        parts.extend(str(term) for term in chunk.get("terms", []))

    for doc in payload.get("docs", []):
        p = Path(doc)
        if not p.is_absolute():
            p = path.parent / p
        if p.exists():
            parts.append(read_text_file(p, max_chars))

    if not parts:
        raise SystemExit(f"{path}: no usable text fields found")

    vector_path = payload.get("out")
    if vector_path:
        vector = Path(vector_path)
        if not vector.is_absolute():
            vector = path.parent / vector
        vector = vector.with_suffix(".f32")
    else:
        vector = path.with_suffix(".f32")
    if not vector.exists():
        sibling = path.with_suffix(".f32")
        if sibling.exists():
            vector = sibling
    if not vector.exists():
        raise SystemExit(f"{path}: no steering vector found beside recipe")
    return "\n".join(parts), vector


def recipe_vector_path(path: Path) -> Path | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not payload.get("good_file"):
        return None
    vector_path = payload.get("out")
    if vector_path:
        vector = Path(vector_path)
        if not vector.is_absolute():
            vector = path.parent / vector
        vector = vector.with_suffix(".f32")
    else:
        vector = path.with_suffix(".f32")
    if not vector.exists():
        sibling = path.with_suffix(".f32")
        if sibling.exists():
            vector = sibling
    return vector if vector.exists() else None


def discover_recipes(items: list[str]) -> list[Path]:
    recipes: list[Path] = []
    for item in items:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            recipes.extend(sorted(
                p for p in path.rglob("*.json")
                if recipe_vector_path(p) is not None
            ))
        elif path.is_file():
            recipes.append(path)
        else:
            raise SystemExit(f"{item}: does not exist")
    deduped = sorted(dict.fromkeys(recipes))
    by_vector: dict[Path, Path] = {}
    for recipe in deduped:
        vector = recipe_vector_path(recipe)
        if vector is None:
            by_vector[recipe.resolve()] = recipe
            continue
        key = vector.resolve()
        old = by_vector.get(key)
        if old is None or recipe.name.endswith(".doc-to-steering.json"):
            by_vector[key] = recipe
    deduped = sorted(by_vector.values())
    if not deduped:
        raise SystemExit("no recipe files found")
    return deduped


def read_vector(path: Path) -> array.array:
    if path.stat().st_size != VECTOR_BYTES:
        raise SystemExit(f"{path}: expected {VECTOR_BYTES} bytes, got {path.stat().st_size}")
    data = array.array("f")
    with path.open("rb") as f:
        data.fromfile(f, VECTOR_FLOATS)
    if len(data) != VECTOR_FLOATS:
        raise SystemExit(f"{path}: expected {VECTOR_FLOATS} floats, got {len(data)}")
    return data


def write_vector(path: Path, data: array.array) -> None:
    if len(data) != VECTOR_FLOATS:
        raise SystemExit(f"refusing to write malformed vector with {len(data)} floats")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        data.tofile(f)


def normalize_layers(data: array.array) -> array.array:
    out = array.array("f", data)
    for layer in range(N_LAYER):
        start = layer * N_EMBD
        end = start + N_EMBD
        n2 = 0.0
        for value in out[start:end]:
            n2 += float(value) * float(value)
        if n2 <= 0.0:
            continue
        inv = 1.0 / math.sqrt(n2)
        for i in range(start, end):
            out[i] = float(out[i]) * inv
    return out


def average_layer_cosine(a: array.array, b: array.array) -> float:
    total = 0.0
    for layer in range(N_LAYER):
        start = layer * N_EMBD
        end = start + N_EMBD
        dot = 0.0
        an = 0.0
        bn = 0.0
        for av, bv in zip(a[start:end], b[start:end]):
            af = float(av)
            bf = float(bv)
            dot += af * bf
            an += af * af
            bn += bf * bf
        if an > 0.0 and bn > 0.0:
            total += dot / math.sqrt(an * bn)
    return total / N_LAYER


def copy_or_reference_vector(src: Path, model_dir: Path, name: str, copy_vectors: bool) -> str:
    if not copy_vectors:
        return str(src)
    basis_dir = model_dir / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    dst = basis_dir / f"{name}.f32"
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(dst.relative_to(model_dir))


def save_model(model_dir: Path, payload: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(model_dir: Path) -> dict:
    model_path = model_dir / "model.json"
    if not model_path.exists():
        raise SystemExit(f"{model_path}: missing model")
    return json.loads(model_path.read_text(encoding="utf-8"))


def resolve_model_file(model_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return model_dir / path


def train(args: argparse.Namespace) -> None:
    recipes = discover_recipes(args.recipe)
    texts: list[str] = []
    vector_paths: list[Path] = []
    raw_counts: list[collections.Counter[int]] = []
    names: list[str] = []

    for i, recipe in enumerate(recipes, 1):
        text, vector_path = load_recipe(recipe, args.max_chars)
        counts = raw_feature_counts(text, args.feature_dim)
        if not counts:
            raise SystemExit(f"{recipe}: text produced no features")
        read_vector(vector_path)
        texts.append(text)
        vector_paths.append(vector_path)
        raw_counts.append(counts)
        names.append(f"ex{i:04d}_{recipe.stem.replace('.', '_')}")

    df: collections.Counter[int] = collections.Counter()
    for counts in raw_counts:
        df.update(counts.keys())
    n = len(raw_counts)
    idf = {idx: math.log((n + 1.0) / (freq + 1.0)) + 1.0 for idx, freq in df.items()}

    model_dir = Path(args.model_dir).expanduser().resolve()
    examples: list[Example] = []
    for name, recipe, vector_path, counts, text in zip(names, recipes, vector_paths, raw_counts, texts):
        vector_file = copy_or_reference_vector(vector_path, model_dir, name, not args.no_copy_vectors)
        features = normalize_sparse(counts, idf)
        examples.append(Example(
            name=name,
            recipe_path=str(recipe),
            vector_file=vector_file,
            features=features,
            text_chars=len(text),
        ))

    payload = {
        "format": "ds4-doc-steering-hypernetwork-v1",
        "kind": args.kind,
        "n_layer": N_LAYER,
        "n_embd": N_EMBD,
        "feature_dim": args.feature_dim,
        "default_top_k": args.top_k,
        "default_temperature": args.temperature,
        "idf": [[idx, value] for idx, value in sorted(idf.items())],
        "examples": [
            {
                "name": ex.name,
                "recipe_path": ex.recipe_path,
                "vector_file": ex.vector_file,
                "features": [[idx, value] for idx, value in sorted(ex.features.items())],
                "text_chars": ex.text_chars,
            }
            for ex in examples
        ],
    }
    if args.kind == "learned-basis":
        weights, bias, losses = train_sparse_softmax(
            [ex.features for ex in examples],
            args.epochs,
            args.lr,
            args.l2,
        )
        payload.update({
            "bias": bias,
            "linear_weights": [
                {"idx": idx, "weights": [value for value in row]}
                for idx, row in sorted(weights.items())
                if any(abs(value) > 1e-12 for value in row)
            ],
            "training": {
                "epochs": args.epochs,
                "lr": args.lr,
                "l2": args.l2,
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
            },
        })
    save_model(model_dir, payload)
    print(f"wrote {model_dir / 'model.json'}")
    if args.kind == "learned-basis":
        first = payload["training"]["loss_first"]
        last = payload["training"]["loss_last"]
        print(f"trained learned-basis hypernetwork with {len(examples)} examples loss {first:0.4f}->{last:0.4f}")
    else:
        print(f"trained retrieval-basis hypernetwork with {len(examples)} examples")


def features_from_docs(docs: list[str], feature_dim: int, idf: dict[int, float], max_chars: int) -> dict[int, float]:
    parts: list[str] = []
    for item in docs:
        path = Path(item).expanduser()
        if path.exists():
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix in {".md", ".txt", ".c", ".h", ".py", ".metal", ".cu", ".m"}:
                        parts.append(read_text_file(child, max_chars))
            else:
                parts.append(read_text_file(path, max_chars))
        else:
            parts.append(item)
    return normalize_sparse(raw_feature_counts("\n".join(parts), feature_dim), idf)


def model_examples(model: dict) -> list[Example]:
    examples: list[Example] = []
    for item in model.get("examples", []):
        examples.append(Example(
            name=item["name"],
            recipe_path=item["recipe_path"],
            vector_file=item["vector_file"],
            features={int(idx): float(value) for idx, value in item["features"]},
            text_chars=int(item.get("text_chars", 0)),
        ))
    if not examples:
        raise SystemExit("model contains no examples")
    return examples


def choose_weights(
    query_features: dict[int, float],
    examples: list[Example],
    top_k: int,
    temperature: float,
    skip_name: str | None = None,
) -> list[tuple[Example, float, float]]:
    scored = [
        (ex, sparse_dot(query_features, ex.features))
        for ex in examples
        if ex.name != skip_name
    ]
    if not scored:
        raise SystemExit("no examples available for prediction")
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:max(1, min(top_k, len(scored)))]
    if temperature <= 0:
        winner, sim = selected[0]
        return [(winner, 1.0, sim)]

    best = selected[0][1]
    exp_scores = [
        math.exp((sim - best) / temperature)
        for _, sim in selected
    ]
    total = sum(exp_scores)
    if total <= 0.0:
        winner, sim = selected[0]
        return [(winner, 1.0, sim)]
    return [
        (ex, value / total, sim)
        for (ex, sim), value in zip(selected, exp_scores)
    ]


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        return []
    temp = max(temperature, 1e-6)
    best = max(logits)
    vals = [math.exp((value - best) / temp) for value in logits]
    total = sum(vals)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in vals]


def train_sparse_softmax(
    features: list[dict[int, float]],
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[dict[int, list[float]], list[float], list[float]]:
    """Train sparse linear text features to basis-vector coefficients."""
    n = len(features)
    weights: dict[int, list[float]] = {}
    bias = [0.0] * n
    losses: list[float] = []
    for _ in range(max(1, epochs)):
        total_loss = 0.0
        for target, x in enumerate(features):
            logits = bias[:]
            for idx, value in x.items():
                row = weights.get(idx)
                if row:
                    for j in range(n):
                        logits[j] += value * row[j]
            probs = softmax(logits)
            total_loss += -math.log(max(probs[target], 1e-12))

            grads = [probs[j] - (1.0 if j == target else 0.0) for j in range(n)]
            for j, grad in enumerate(grads):
                bias[j] -= lr * grad
            for idx, value in x.items():
                row = weights.setdefault(idx, [0.0] * n)
                for j, grad in enumerate(grads):
                    update = grad * value
                    if l2:
                        update += l2 * row[j]
                    row[j] -= lr * update
        losses.append(total_loss / n)
    return weights, bias, losses


def learned_logits(query_features: dict[int, float], model: dict) -> list[float]:
    logits = [float(value) for value in model.get("bias", [])]
    rows = {
        int(item["idx"]): [float(value) for value in item["weights"]]
        for item in model.get("linear_weights", [])
    }
    for idx, value in query_features.items():
        row = rows.get(idx)
        if row:
            for j in range(len(logits)):
                logits[j] += value * row[j]
    return logits


def choose_learned_weights(
    query_features: dict[int, float],
    examples: list[Example],
    model: dict,
    top_k: int,
    temperature: float,
    skip_name: str | None = None,
) -> list[tuple[Example, float, float]]:
    logits = learned_logits(query_features, model)
    if len(logits) != len(examples):
        raise SystemExit("learned model output size does not match examples")
    probs = softmax(logits, temperature)
    scored = [
        (ex, probs[i], logits[i])
        for i, ex in enumerate(examples)
        if ex.name != skip_name
    ]
    if not scored:
        raise SystemExit("no examples available for learned prediction")
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:max(1, min(top_k, len(scored)))]
    total = sum(weight for _, weight, _ in selected)
    if total <= 0.0:
        winner, _, logit = selected[0]
        return [(winner, 1.0, logit)]
    return [(ex, weight / total, logit) for ex, weight, logit in selected]


def choose_model_weights(
    query_features: dict[int, float],
    examples: list[Example],
    model: dict,
    top_k: int,
    temperature: float,
    skip_name: str | None = None,
) -> list[tuple[Example, float, float]]:
    if model.get("kind") == "learned-basis":
        return choose_learned_weights(query_features, examples, model, top_k, temperature, skip_name)
    return choose_weights(query_features, examples, top_k, temperature, skip_name)


def mix_vectors(model_dir: Path, weighted: list[tuple[Example, float, float]]) -> array.array:
    out = array.array("f", [0.0]) * VECTOR_FLOATS
    for ex, weight, _ in weighted:
        vec = read_vector(resolve_model_file(model_dir, ex.vector_file))
        for i, value in enumerate(vec):
            out[i] += float(weight) * float(value)
    return normalize_layers(out)


def predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser().resolve()
    model = load_model(model_dir)
    idf = {int(idx): float(value) for idx, value in model.get("idf", [])}
    examples = model_examples(model)
    feature_dim = int(model["feature_dim"])
    query = features_from_docs(args.doc, feature_dim, idf, args.max_chars)
    if not query:
        raise SystemExit("prediction text produced no features")
    weighted = choose_model_weights(
        query,
        examples,
        model,
        args.top_k or int(model.get("default_top_k", 4)),
        args.temperature if args.temperature is not None else float(model.get("default_temperature", 0.08)),
    )
    out = Path(args.out).expanduser().resolve()
    vector = mix_vectors(model_dir, weighted)
    write_vector(out, vector)
    meta = {
        "format": "ds4-doc-steering-hypernetwork-prediction-v1",
        "model_dir": str(model_dir),
        "out": str(out),
        "doc": args.doc,
        "weights": [
            {"name": ex.name, "weight": weight, "basis_score": sim}
            for ex, weight, sim in weighted
        ],
        "note": "Use this .f32 with --dir-steering-file; negative FFN scales amplify the predicted document register.",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    for ex, weight, sim in weighted:
        print(f"{weight:0.3f}  basis_score={sim:0.3f}  {ex.name}")


def evaluate(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser().resolve()
    model = load_model(model_dir)
    examples = model_examples(model)
    top_k = args.top_k or int(model.get("default_top_k", 4))
    temperature = args.temperature if args.temperature is not None else float(model.get("default_temperature", 0.08))

    rows: list[tuple[str, float, str]] = []
    for ex in examples:
        query = ex.features
        weighted = choose_model_weights(
            query,
            examples,
            model,
            top_k,
            temperature,
            skip_name=ex.name if args.leave_one_out else None,
        )
        pred = mix_vectors(model_dir, weighted)
        gold = read_vector(resolve_model_file(model_dir, ex.vector_file))
        score = average_layer_cosine(pred, gold)
        rows.append((ex.name, score, weighted[0][0].name))

    mean = sum(score for _, score, _ in rows) / len(rows)
    print(f"examples={len(rows)} mean_layer_cosine={mean:0.4f}")
    for name, score, top in rows:
        print(f"{score:0.4f}  top={top}  {name}")
    if args.min_mean_cosine is not None and mean < args.min_mean_cosine:
        raise SystemExit(f"mean cosine {mean:0.4f} < required {args.min_mean_cosine:0.4f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train/predict doc-conditioned DS4 steering vectors.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    train_ap = sub.add_parser("train")
    train_ap.add_argument("--recipe", action="append", required=True,
                          help="recipe JSON or directory; may be repeated")
    train_ap.add_argument("--model-dir", required=True)
    train_ap.add_argument("--feature-dim", type=int, default=4096)
    train_ap.add_argument("--max-chars", type=int, default=200_000)
    train_ap.add_argument("--top-k", type=int, default=4)
    train_ap.add_argument("--temperature", type=float, default=0.08)
    train_ap.add_argument("--kind", default="learned-basis",
                          choices=("learned-basis", "retrieval-basis"))
    train_ap.add_argument("--epochs", type=int, default=160)
    train_ap.add_argument("--lr", type=float, default=0.25)
    train_ap.add_argument("--l2", type=float, default=0.0001)
    train_ap.add_argument("--no-copy-vectors", action="store_true")
    train_ap.set_defaults(func=train)

    pred_ap = sub.add_parser("predict")
    pred_ap.add_argument("--model-dir", required=True)
    pred_ap.add_argument("--doc", action="append", required=True,
                         help="document path or literal text; may be repeated")
    pred_ap.add_argument("--out", required=True)
    pred_ap.add_argument("--top-k", type=int, default=0)
    pred_ap.add_argument("--temperature", type=float, default=None)
    pred_ap.add_argument("--max-chars", type=int, default=200_000)
    pred_ap.set_defaults(func=predict)

    eval_ap = sub.add_parser("eval")
    eval_ap.add_argument("--model-dir", required=True)
    eval_ap.add_argument("--top-k", type=int, default=0)
    eval_ap.add_argument("--temperature", type=float, default=None)
    eval_ap.add_argument("--leave-one-out", action="store_true")
    eval_ap.add_argument("--min-mean-cosine", type=float, default=None)
    eval_ap.set_defaults(func=evaluate)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
