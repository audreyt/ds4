#!/usr/bin/env python3
"""Context-augmented generation for DS4 lore.

This is the factual-memory companion to directional steering:

    repo/docs -> lore pack -> query retrieval -> cited prompt -> ds4

Steering can bias register and answer posture.  The lore prompt supplies exact
facts, file names, commands, and source anchors so the model has something real
to cite instead of inventing repository lore.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTS = {
    ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".inc", ".m", ".metal",
    ".md", ".py", ".sh", ".txt", ".json", ".jsonl",
}

DEFAULT_EXCLUDES = {
    ".git",
    ".claude",
    ".direnv",
    ".venv",
    "__pycache__",
    "dir-steering/out",
    "gguf",
    "node_modules",
    "tmp",
}

STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "before", "being", "bool", "but", "can", "char", "const", "default",
    "does", "each", "else", "false", "file", "float", "for", "from",
    "have", "help", "here", "into", "long", "make", "more", "must", "not",
    "one", "only", "other", "path", "read", "return", "same", "should",
    "static", "str", "struct", "that", "the", "then", "there", "this",
    "true", "uint32_t", "uint64_t", "use", "used", "using", "void", "when",
    "where", "with", "would",
}

CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+$")
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")

QUERY_EXPANSIONS = {
    "build": ["capture", "generate", "build_direction", "build_doc_direction"],
    "test": ["evaluate", "evaluating", "sweep", "run_sweep", "scales", "prompts"],
    "steering": ["directional", "direction", "dir_steering"],
    "vector": ["direction", "f32", "vectors"],
    "script": ["tool", "tools", "py"],
    "audrey": ["唐鳳"],
    "tang": ["唐鳳"],
    "civic": ["公民", "civic"],
    "democracy": ["民主", "democratic"],
}

TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]{2,}|[0-9]{2,}|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}"
)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(")
SPACE_RE = re.compile(r"[ \t]+")


@dataclass
class LoreRecord:
    id: str
    path: str
    chunk: int
    anchor: str
    terms: list[str]
    text: str
    source_date: str = ""
    title: str = ""
    score: float = 0.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_exts(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_EXTS)
    out: set[str] = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            out.add(item if item.startswith(".") else f".{item}")
    return out


def is_excluded(path: Path, root: Path, excludes: set[str]) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    parts = set(path.parts)
    for item in excludes:
        item = item.strip("/")
        if not item:
            continue
        if item in parts or rel == item or rel.startswith(f"{item}/"):
            return True
    return False


def source_date(path: Path) -> str:
    match = DATE_PREFIX_RE.match(path.name)
    return match.group(1) if match else ""


def source_title(path: Path, text: str = "") -> str:
    match = HEADING_RE.search(text)
    if match:
        return flatten(match.group(1), 160)
    stem = DATE_PREFIX_RE.sub("", path.stem)
    return stem.replace("-", " ").replace("_", " ").strip()


def date_in_range(value: str, after: str, before: str) -> bool:
    if after and (not value or value < after):
        return False
    if before and (not value or value > before):
        return False
    return True


def iter_paths(inputs: list[str], root: Path, exts: set[str], excludes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.exists():
            raise SystemExit(f"{raw}: does not exist")
        if path.is_file():
            if path.suffix in exts and not is_excluded(path, root, excludes):
                paths.append(path)
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix in exts and not is_excluded(candidate, root, excludes):
                paths.append(candidate.resolve())
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no lore files found")
    return paths


def filter_paths_by_date(paths: list[Path], after: str, before: str) -> list[Path]:
    if not after and not before:
        return paths
    filtered = [path for path in paths if date_in_range(source_date(path), after, before)]
    if not filtered:
        raise SystemExit(f"no dated lore files found in range after={after or '*'} before={before or '*'}")
    return filtered


def read_text(path: Path, max_chars: int) -> str:
    text = path.read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ")
    if max_chars > 0 and len(text) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        text = text[:head] + "\n\n[... middle omitted by lore_cag.py ...]\n\n" + text[-tail:]
    return text


def split_chunks(text: str, max_chars: int, min_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        body = "\n".join(cur).strip()
        if len(body) >= min_chars:
            chunks.append(body)
        cur = []
        cur_len = 0

    for line in text.splitlines():
        line = line.rstrip()
        add_len = len(line) + 1
        starts_section = bool(HEADING_RE.match(line)) and cur_len >= min_chars
        if cur and (starts_section or cur_len + add_len > max_chars):
            flush()
        cur.append(line)
        cur_len += add_len
    flush()

    if not chunks and text.strip():
        body = text.strip()
        for start in range(0, len(body), max_chars):
            chunk = body[start:start + max_chars].strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
    return chunks


def flatten(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0].strip()
    return f"{cut} ..."


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        if CJK_RE.match(raw):
            if len(raw) <= 16:
                tokens.append(raw)
            for n in (2, 3, 4):
                if len(raw) >= n:
                    tokens.extend(raw[i:i + n] for i in range(0, len(raw) - n + 1))
            continue
        token = raw.lower()
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if len(part) > 2 and part not in STOPWORDS)
    return tokens


def term_counts(text: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for token in tokenize(text):
        if token in STOPWORDS:
            continue
        weight = 1
        if "_" in token:
            weight += 3
        if token.startswith(("ds4", "dsv4")):
            weight += 4
        if token.isupper() and len(token) > 3:
            weight += 1
        if CJK_RE.match(token):
            weight += 1
        counts[token] += weight
    return counts


def choose_anchor(text: str, terms: list[str], fallback: str) -> str:
    headings = [flatten(h, 90) for h in HEADING_RE.findall(text)]
    if headings:
        return headings[0]
    funcs = [name for name in FUNC_RE.findall(text) if name.lower() not in STOPWORDS]
    if funcs:
        return funcs[0]
    if terms:
        return ", ".join(terms[:3])
    return fallback


def record_from_payload(payload: dict) -> LoreRecord:
    return LoreRecord(
        id=str(payload["id"]),
        path=str(payload["path"]),
        chunk=int(payload["chunk"]),
        anchor=str(payload.get("anchor", "")),
        terms=[str(term) for term in payload.get("terms", [])],
        text=str(payload["text"]),
        source_date=str(payload.get("source_date", "")),
        title=str(payload.get("title", "")),
        score=float(payload.get("score", 0.0)),
    )


def record_to_payload(record: LoreRecord) -> dict:
    payload = {
        "id": record.id,
        "path": record.path,
        "chunk": record.chunk,
        "anchor": record.anchor,
        "terms": record.terms,
        "text": record.text,
    }
    if record.source_date:
        payload["source_date"] = record.source_date
    if record.title:
        payload["title"] = record.title
    return payload


def load_pack(path: Path) -> list[LoreRecord]:
    records: list[LoreRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("format") == "ds4-lore-pack-v1":
                continue
            try:
                records.append(record_from_payload(payload))
            except KeyError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed lore record missing {exc}") from exc
    if not records:
        raise SystemExit(f"{path}: lore pack is empty")
    return records


def pack_cmd(args: argparse.Namespace) -> None:
    root = repo_root()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    excludes = set(DEFAULT_EXCLUDES)
    excludes.update(args.exclude)
    paths = iter_paths(args.doc, root, parse_exts(args.include_ext), excludes)
    paths = filter_paths_by_date(paths, args.after, args.before)

    records: list[LoreRecord] = []
    for path in paths:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        text = read_text(path, args.max_file_chars)
        date = source_date(path)
        title = source_title(path, text)
        for chunk_index, chunk_text in enumerate(split_chunks(text, args.max_chunk_chars, args.min_chunk_chars), 1):
            terms = [term for term, _ in term_counts(chunk_text).most_common(args.terms)]
            anchor = choose_anchor(chunk_text, terms, Path(rel).stem)
            record_id = f"{len(records) + 1:06d}"
            records.append(LoreRecord(
                id=record_id,
                path=rel,
                chunk=chunk_index,
                anchor=anchor,
                terms=terms,
                text=chunk_text,
                source_date=date,
                title=title,
            ))
            if args.max_records and len(records) >= args.max_records:
                break
        if args.max_records and len(records) >= args.max_records:
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "format": "ds4-lore-pack-v1",
            "root": str(root),
            "record_count": len(records),
            "source_count": len(paths),
            "after": args.after,
            "before": args.before,
        }) + "\n")
        for record in records:
            f.write(json.dumps(record_to_payload(record), ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    print(f"packed {len(records)} chunks from {len(paths)} files")


def score_record(query_tokens: collections.Counter[str], query_text: str, record: LoreRecord) -> float:
    haystack = f"{record.path}\n{record.source_date}\n{record.title}\n{record.anchor}\n{' '.join(record.terms)}\n{record.text}"
    counts = collections.Counter(tokenize(haystack))
    if not counts:
        return 0.0

    score = 0.0
    for token, q_count in query_tokens.items():
        tf = counts.get(token, 0)
        if tf:
            score += (1.0 + math.log(tf)) * (1.0 + math.log(q_count))
            if token in {term.lower() for term in record.terms}:
                score += 1.5
            if token in record.path.lower():
                score += 1.0

    for phrase in re.findall(r"`([^`]+)`", query_text):
        phrase = phrase.strip().lower()
        if phrase and phrase in haystack.lower():
            score += 8.0

    # Prefer compact chunks when the lexical evidence is otherwise similar.
    return score / (1.0 + len(record.text) / 8000.0)


def token_set(record: LoreRecord) -> set[str]:
    return set(tokenize(f"{record.path}\n{record.title}\n{record.anchor}\n{' '.join(record.terms)}\n{record.text}"))


def overlap_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def mmr_select(scored: list[LoreRecord], top_k: int, mmr_lambda: float) -> list[LoreRecord]:
    if top_k <= 0 or not scored:
        return []
    if mmr_lambda >= 0.999:
        return scored[:top_k]
    selected: list[LoreRecord] = []
    selected_terms: list[set[str]] = []
    pool = scored[:max(top_k * 25, top_k)]
    score_scale = max(1e-9, pool[0].score if pool else 1.0)
    cache: dict[str, set[str]] = {}
    while pool and len(selected) < top_k:
        best_i = 0
        best_value = -1e18
        for i, record in enumerate(pool):
            terms = cache.setdefault(record.id, token_set(record))
            novelty_penalty = max((overlap_similarity(terms, old) for old in selected_terms), default=0.0)
            value = mmr_lambda * (record.score / score_scale) - (1.0 - mmr_lambda) * novelty_penalty
            if value > best_value:
                best_value = value
                best_i = i
        chosen = pool.pop(best_i)
        selected.append(chosen)
        selected_terms.append(cache.setdefault(chosen.id, token_set(chosen)))
    return selected


def expand_neighbor_records(selected: list[LoreRecord], all_records: list[LoreRecord], neighbor_chunks: int) -> list[LoreRecord]:
    if neighbor_chunks <= 0:
        return selected
    by_key = {(record.path, record.chunk): record for record in all_records}
    out: list[LoreRecord] = []
    seen: set[tuple[str, int]] = set()
    for record in selected:
        for chunk in range(record.chunk - neighbor_chunks, record.chunk + neighbor_chunks + 1):
            key = (record.path, chunk)
            if key in seen:
                continue
            neighbor = by_key.get(key)
            if not neighbor:
                continue
            distance = abs(chunk - record.chunk)
            score = record.score if distance == 0 else record.score * max(0.25, 0.72 ** distance)
            out.append(LoreRecord(
                id=neighbor.id,
                path=neighbor.path,
                chunk=neighbor.chunk,
                anchor=neighbor.anchor,
                terms=neighbor.terms,
                text=neighbor.text,
                source_date=neighbor.source_date,
                title=neighbor.title,
                score=score,
            ))
            seen.add(key)
    return out


def retrieve_records_from_list(
    records: list[LoreRecord],
    query: str,
    top_k: int,
    min_score: float = 0.0,
    after: str = "",
    before: str = "",
    neighbor_chunks: int = 0,
    mmr_lambda: float = 0.9,
) -> list[LoreRecord]:
    query_tokens = collections.Counter(tokenize(query))
    for token, count in list(query_tokens.items()):
        for expanded in QUERY_EXPANSIONS.get(token, []):
            query_tokens[expanded] += max(1, int(math.ceil(count * 0.5)))
    if not query_tokens:
        raise SystemExit("query produced no retrieval tokens")
    filtered_records = [
        record for record in records
        if date_in_range(record.source_date, after, before)
    ]
    if not filtered_records:
        raise SystemExit(f"no lore records match date filters after={after or '*'} before={before or '*'}")
    scored: list[LoreRecord] = []
    for record in filtered_records:
        score = score_record(query_tokens, query, record)
        if score >= min_score:
            scored.append(LoreRecord(
                id=record.id,
                path=record.path,
                chunk=record.chunk,
                anchor=record.anchor,
                terms=record.terms,
                text=record.text,
                source_date=record.source_date,
                title=record.title,
                score=score,
            ))
    scored.sort(key=lambda item: item.score, reverse=True)
    selected = mmr_select(scored, top_k, max(0.0, min(1.0, mmr_lambda)))
    return expand_neighbor_records(selected, filtered_records, neighbor_chunks)


def retrieve_records(
    pack: Path,
    query: str,
    top_k: int,
    min_score: float = 0.0,
    after: str = "",
    before: str = "",
    neighbor_chunks: int = 0,
    mmr_lambda: float = 0.9,
) -> list[LoreRecord]:
    return retrieve_records_from_list(
        load_pack(pack),
        query,
        top_k,
        min_score,
        after,
        before,
        neighbor_chunks,
        mmr_lambda,
    )


def trim_excerpt(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0].rstrip()
    if len(cut) < limit * 0.5:
        cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{cut}\n[... excerpt trimmed ...]"


def cap_total(records: list[LoreRecord], max_total_chars: int, excerpt_chars: int) -> list[LoreRecord]:
    out: list[LoreRecord] = []
    used = 0
    for record in records:
        text = trim_excerpt(record.text, excerpt_chars)
        if max_total_chars > 0 and used + len(text) > max_total_chars:
            remaining = max_total_chars - used
            if remaining < 300:
                break
            text = trim_excerpt(text, remaining)
        out.append(LoreRecord(
            id=record.id,
            path=record.path,
            chunk=record.chunk,
            anchor=record.anchor,
            terms=record.terms,
            text=text,
            source_date=record.source_date,
            title=record.title,
            score=record.score,
        ))
        used += len(text)
        if max_total_chars > 0 and used >= max_total_chars:
            break
    return out


def retrieve_for_prompt(args: argparse.Namespace) -> list[LoreRecord]:
    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = repo_root() / pack
    records = retrieve_records(
        pack,
        args.query,
        args.top_k,
        args.min_score,
        args.after,
        args.before,
        args.neighbor_chunks,
        args.mmr_lambda,
    )
    return cap_total(records, args.max_context_chars, args.excerpt_chars)


def retrieve_cmd(args: argparse.Namespace) -> None:
    records = retrieve_for_prompt(args)
    payload = {
        "format": "ds4-lore-retrieval-v1",
        "query": args.query,
        "records": [
            {
                **record_to_payload(record),
                "score": record.score,
            }
            for record in records
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


def compose_prompt(query: str, records: list[LoreRecord]) -> str:
    pieces = [
        "You are answering from a DS4 lore pack.",
        "Use only the cited lore excerpts for concrete repository facts, commands, file names, and APIs.",
        "When you state a concrete fact, cite the bracketed source number such as [1].",
        "If the excerpts do not contain the fact, say you do not see it in the provided lore.",
        "",
        "<lore>",
    ]
    for i, record in enumerate(records, 1):
        source_bits = [
            f"file: {record.path}",
            f"chunk: {record.chunk}",
        ]
        if record.source_date:
            source_bits.append(f"date: {record.source_date}")
        if record.title:
            source_bits.append(f"title: {record.title}")
        source_bits.extend([
            f"anchor: {record.anchor}",
            f"score: {record.score:.3f}",
        ])
        pieces.extend([
            f"[{i}] {' '.join(source_bits)}",
            "```text",
            record.text.strip(),
            "```",
            "",
        ])
    pieces.extend([
        "</lore>",
        "",
        f"Question: {query}",
        "",
        "Answer with concise DS4-maintainer judgment. Prefer exact commands and paths only when present in the lore excerpts.",
    ])
    return "\n".join(pieces)


def prompt_cmd(args: argparse.Namespace) -> None:
    records = retrieve_for_prompt(args)
    prompt = compose_prompt(args.query, records)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(prompt)


def predict_steering(args: argparse.Namespace, records: list[LoreRecord], temp_dir: Path) -> Path:
    hnet = Path(args.hnet_model_dir)
    if not hnet.is_absolute():
        hnet = repo_root() / hnet
    steering_out = Path(args.hnet_out) if args.hnet_out else temp_dir / "predicted_lore_steering.f32"
    if not steering_out.is_absolute():
        steering_out = repo_root() / steering_out
    steering_doc = temp_dir / "retrieved_lore_for_hnet.txt"
    steering_doc.write_text("\n\n".join([args.query] + [record.text for record in records]), encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("doc_steering_hypernetwork.py")),
        "predict",
        "--model-dir", str(hnet),
        "--doc", str(steering_doc),
        "--out", str(steering_out),
        "--top-k", str(args.hnet_top_k),
        "--temperature", str(args.hnet_temperature),
    ]
    subprocess.run(cmd, cwd=repo_root(), check=True)
    return steering_out


def run_cmd(args: argparse.Namespace) -> None:
    records = retrieve_for_prompt(args)
    prompt = compose_prompt(args.query, records)

    tmp = tempfile.TemporaryDirectory(prefix="ds4-lore-cag-")
    temp_dir = Path(tmp.name)
    prompt_path = Path(args.out_prompt) if args.out_prompt else temp_dir / "prompt.txt"
    if not prompt_path.is_absolute():
        prompt_path = repo_root() / prompt_path
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    steering_file = args.dir_steering_file
    if args.hnet_model_dir:
        steering_file = str(predict_steering(args, records, temp_dir))

    cmd = [
        args.ds4,
        "-m", args.model,
        "--ctx", str(args.ctx),
        "--prompt-file", str(prompt_path),
        "-n", str(args.tokens),
        "--temp", str(args.temp),
        "-sys", args.system,
    ]
    if args.nothink:
        cmd.append("--nothink")
    else:
        cmd.append("--think")
    if steering_file:
        cmd.extend([
            "--dir-steering-file", steering_file,
            "--dir-steering-ffn", str(args.dir_steering_ffn),
            "--dir-steering-attn", str(args.dir_steering_attn),
        ])

    if args.print_command or args.dry_run:
        print(shlex.join(cmd), flush=True)
        print(f"prompt: {prompt_path}", flush=True)
        if steering_file:
            print(f"steering: {steering_file}", flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=repo_root(), check=True)

    if not args.dry_run:
        tmp.cleanup()


def add_retrieval_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pack", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--excerpt-chars", type=int, default=1800)
    ap.add_argument("--max-context-chars", type=int, default=9000)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--after", default="", help="retrieve only sources dated on or after YYYY-MM-DD")
    ap.add_argument("--before", default="", help="retrieve only sources dated on or before YYYY-MM-DD")
    ap.add_argument("--neighbor-chunks", type=int, default=0,
                    help="also include this many adjacent chunks around each selected chunk")
    ap.add_argument("--mmr-lambda", type=float, default=0.9,
                    help="MMR relevance/diversity tradeoff; 1.0 is pure score ranking")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build and query DS4 context-augmented lore packs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pack_ap = sub.add_parser("pack")
    pack_ap.add_argument("--doc", action="append", required=True,
                         help="source file or directory; may be repeated")
    pack_ap.add_argument("--out", required=True)
    pack_ap.add_argument("--include-ext", default="")
    pack_ap.add_argument("--exclude", action="append", default=[])
    pack_ap.add_argument("--max-file-chars", type=int, default=300_000)
    pack_ap.add_argument("--max-chunk-chars", type=int, default=2400)
    pack_ap.add_argument("--min-chunk-chars", type=int, default=220)
    pack_ap.add_argument("--terms", type=int, default=12)
    pack_ap.add_argument("--max-records", type=int, default=0)
    pack_ap.add_argument("--after", default="", help="include only sources dated on or after YYYY-MM-DD")
    pack_ap.add_argument("--before", default="", help="include only sources dated on or before YYYY-MM-DD")
    pack_ap.set_defaults(func=pack_cmd)

    retrieve_ap = sub.add_parser("retrieve")
    add_retrieval_args(retrieve_ap)
    retrieve_ap.add_argument("--out", default="")
    retrieve_ap.set_defaults(func=retrieve_cmd)

    prompt_ap = sub.add_parser("prompt")
    add_retrieval_args(prompt_ap)
    prompt_ap.add_argument("--out", default="")
    prompt_ap.set_defaults(func=prompt_cmd)

    run_ap = sub.add_parser("run")
    add_retrieval_args(run_ap)
    run_ap.add_argument("--ds4", default="./ds4")
    run_ap.add_argument("--model", default="ds4flash.gguf")
    run_ap.add_argument("--ctx", type=int, default=32768)
    run_ap.add_argument("--tokens", type=int, default=500)
    run_ap.add_argument("--temp", type=float, default=0.0)
    run_ap.add_argument("--system", default="You are a precise DS4 repository assistant.")
    mode = run_ap.add_mutually_exclusive_group()
    mode.add_argument("--nothink", dest="nothink", action="store_true", default=True)
    mode.add_argument("--think", dest="nothink", action="store_false")
    run_ap.add_argument("--out-prompt", default="")
    run_ap.add_argument("--dir-steering-file", default="")
    run_ap.add_argument("--dir-steering-ffn", type=float, default=-0.5)
    run_ap.add_argument("--dir-steering-attn", type=float, default=0.0)
    run_ap.add_argument("--hnet-model-dir", default="")
    run_ap.add_argument("--hnet-out", default="")
    run_ap.add_argument("--hnet-top-k", type=int, default=1)
    run_ap.add_argument("--hnet-temperature", type=float, default=0.0)
    run_ap.add_argument("--dry-run", action="store_true")
    run_ap.add_argument("--print-command", action="store_true")
    run_ap.set_defaults(func=run_cmd)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
