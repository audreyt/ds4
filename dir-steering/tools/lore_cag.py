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
import html
import json
import math
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
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

SQLITE_INDEX_FORMAT = "ds4-lore-sqlite-index-v1"
SQLITE_INDEX_AUTOBUILD_BYTES = 32 * 1024 * 1024
SQLITE_INDEX_TOKEN_LIMIT = 192
SQLITE_MAX_VARS = 900
SECTION_INDEX_FORMAT = "ds4-archive-section-index-v1"
PROMPT_SOURCE_RE = re.compile(
    r"^\[(\d+)\]\s+file:\s+(.*?)\s+chunk:\s+(\d+)"
    r"(?:\s+date:\s+(.*?))?"
    r"(?:\s+title:\s+(.*?))?"
    r"\s+anchor:\s+(.*?)\s+score:\s+([0-9.]+)",
    re.MULTILINE,
)
HTML_BLOCK_RE = re.compile(r"<(script|style)\b[\s\S]*?</\1>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


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


@dataclass
class PromptSource:
    index: int
    path: str
    chunk: int
    title: str
    text: str = ""
    section_id: int | None = None
    section_speaker: str = ""


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


def iter_pack_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("format") == "ds4-lore-pack-v1":
                continue
            try:
                yield record_from_payload(payload)
            except KeyError as exc:
                raise SystemExit(f"{path}:{lineno}: malformed lore record missing {exc}") from exc


def load_pack(path: Path) -> list[LoreRecord]:
    records = list(iter_pack_records(path))
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


def record_haystack(record: LoreRecord) -> str:
    return f"{record.path}\n{record.source_date}\n{record.title}\n{record.anchor}\n{' '.join(record.terms)}\n{record.text}"


def score_record(query_tokens: collections.Counter[str], query_text: str, record: LoreRecord) -> float:
    haystack = record_haystack(record)
    counts = collections.Counter(tokenize(haystack))
    if not counts:
        return 0.0
    return score_counts(query_tokens, query_text, record, counts, haystack.lower())


def score_counts(
    query_tokens: collections.Counter[str],
    query_text: str,
    record: LoreRecord,
    counts: collections.Counter[str],
    haystack_lower: str,
) -> float:
    score = 0.0
    term_lowers = {term.lower() for term in record.terms}
    for token, q_count in query_tokens.items():
        tf = counts.get(token, 0)
        if tf:
            score += (1.0 + math.log(tf)) * (1.0 + math.log(q_count))
            if token in term_lowers:
                score += 1.5
            if token in record.path.lower():
                score += 1.0

    for phrase in re.findall(r"`([^`]+)`", query_text):
        phrase = phrase.strip().lower()
        if phrase and phrase in haystack_lower:
            score += 8.0

    # Prefer compact chunks when the lexical evidence is otherwise similar.
    return score / (1.0 + len(record.text) / 8000.0)


class LoreIndex:
    """In-memory inverted index for repeated retrieval over large lore packs."""

    def __init__(self, records: list[LoreRecord]):
        self.records = records
        self.counts: list[collections.Counter[str]] = []
        self.haystack_lowers: list[str] = []
        self.postings: dict[str, list[int]] = collections.defaultdict(list)
        for i, record in enumerate(records):
            haystack = record_haystack(record)
            counts = collections.Counter(tokenize(haystack))
            self.counts.append(counts)
            self.haystack_lowers.append(haystack.lower())
            for token in counts:
                self.postings[token].append(i)

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float = 0.0,
        after: str = "",
        before: str = "",
        neighbor_chunks: int = 0,
        mmr_lambda: float = 0.9,
    ) -> list[LoreRecord]:
        query_tokens = expanded_query_tokens(query)
        candidates = self.candidate_indexes(query, query_tokens, top_k)

        scored: list[LoreRecord] = []
        for idx in candidates:
            record = self.records[idx]
            if not date_in_range(record.source_date, after, before):
                continue
            score = score_counts(query_tokens, query, record, self.counts[idx], self.haystack_lowers[idx])
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
        filtered_records = [
            record for record in self.records
            if date_in_range(record.source_date, after, before)
        ]
        return expand_neighbor_records(selected, filtered_records, neighbor_chunks)

    def candidate_indexes(self, query: str, query_tokens: collections.Counter[str], top_k: int) -> set[int]:
        phrase_tokens: collections.Counter[str] = collections.Counter()
        for phrase in re.findall(r"`([^`]+)`", query):
            phrase_tokens.update(tokenize(phrase))

        base_tokens = phrase_tokens or query_tokens
        token_postings = [
            (token, self.postings[token])
            for token in base_tokens
            if token in self.postings
        ]
        if not token_postings:
            return set(range(len(self.records)))

        token_postings.sort(key=lambda item: len(item[1]))
        min_candidates = max(top_k * 30, 80)
        candidates = set(token_postings[0][1])

        # For quoted needle queries, intersect rare phrase terms aggressively.
        # For open-ended queries, keep enough recall by stopping before the pool
        # collapses too far.
        for _, postings in token_postings[1:8]:
            narrowed = candidates & set(postings)
            if phrase_tokens:
                if narrowed:
                    candidates = narrowed
            elif len(narrowed) >= min_candidates:
                candidates = narrowed

        if len(candidates) < max(top_k, 3) and len(token_postings) > 1:
            fallback: set[int] = set()
            for _, postings in token_postings[:8]:
                fallback.update(postings)
            if fallback:
                candidates = fallback
        return candidates


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
    return LoreIndex(records).retrieve(query, top_k, min_score, after, before, neighbor_chunks, mmr_lambda)


def default_sqlite_index_path(pack: Path) -> Path:
    return pack.with_name(f"{pack.name}.sqlite")


def resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path


def sqlite_index_meta(pack: Path) -> dict[str, str]:
    stat = pack.stat()
    return {
        "format": SQLITE_INDEX_FORMAT,
        "pack": str(pack.resolve()),
        "pack_size": str(stat.st_size),
        "pack_mtime_ns": str(stat.st_mtime_ns),
        "token_limit": str(SQLITE_INDEX_TOKEN_LIMIT),
    }


def sqlite_index_is_fresh(index_path: Path, pack: Path) -> bool:
    if not index_path.exists():
        return False
    expected = sqlite_index_meta(pack)
    try:
        conn = sqlite3.connect(index_path)
        rows = dict(conn.execute("SELECT key, value FROM meta"))
        conn.close()
    except sqlite3.Error:
        return False
    return all(rows.get(key) == value for key, value in expected.items())


def index_tokens_for_record(record: LoreRecord) -> list[str]:
    counts = term_counts(record_haystack(record))
    metadata_counts = term_counts(f"{record.path}\n{record.source_date}\n{record.title}\n{record.anchor}\n{' '.join(record.terms)}")
    for token, count in metadata_counts.items():
        counts[token] += count + 6

    selected: list[str] = []
    seen: set[str] = set()
    priority_text = f"{record.path}\n{record.source_date}\n{record.title}\n{record.anchor}\n{' '.join(record.terms)}"
    for token in tokenize(priority_text):
        if token not in seen:
            selected.append(token)
            seen.add(token)
    for token, _ in counts.most_common(SQLITE_INDEX_TOKEN_LIMIT):
        if token not in seen:
            selected.append(token)
            seen.add(token)
        if len(selected) >= SQLITE_INDEX_TOKEN_LIMIT:
            break
    return selected


def build_sqlite_index(pack: Path, index_path: Path, rebuild: bool = False) -> None:
    if not rebuild and sqlite_index_is_fresh(index_path, pack):
        print(f"lore index is current: {index_path}", file=sys.stderr)
        return

    print(f"building lore index {index_path} from {pack}", file=sys.stderr)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(f"{index_path.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    conn: sqlite3.Connection | None = None
    count = 0
    try:
        conn = sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.executescript("""
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE records (
                rid INTEGER PRIMARY KEY,
                id TEXT NOT NULL,
                path TEXT NOT NULL,
                chunk INTEGER NOT NULL,
                anchor TEXT NOT NULL,
                terms TEXT NOT NULL,
                source_date TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE postings (
                token TEXT NOT NULL,
                rid INTEGER NOT NULL
            );
        """)
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            sorted(sqlite_index_meta(pack).items()),
        )

        record_rows: list[tuple[int, str, str, int, str, str, str, str, str]] = []
        posting_rows: list[tuple[str, int]] = []

        def flush() -> None:
            nonlocal record_rows, posting_rows
            if record_rows:
                conn.executemany(
                    """
                    INSERT INTO records(rid, id, path, chunk, anchor, terms, source_date, title, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record_rows,
                )
                record_rows = []
            if posting_rows:
                conn.executemany("INSERT INTO postings(token, rid) VALUES (?, ?)", posting_rows)
                posting_rows = []

        for record in iter_pack_records(pack):
            count += 1
            rid = count
            record_rows.append((
                rid,
                record.id,
                record.path,
                record.chunk,
                record.anchor,
                json.dumps(record.terms, ensure_ascii=False),
                record.source_date,
                record.title,
                record.text,
            ))
            posting_rows.extend((token, rid) for token in index_tokens_for_record(record))
            if count % 1000 == 0:
                flush()
            if count % 25000 == 0:
                print(f"indexed {count} lore chunks...", file=sys.stderr)
        flush()
        if count == 0:
            raise SystemExit(f"{pack}: lore pack is empty")
        conn.execute("CREATE INDEX postings_token_idx ON postings(token)")
        conn.execute("CREATE INDEX records_path_chunk_idx ON records(path, chunk)")
        conn.commit()
        conn.close()
        conn = None
        tmp.replace(index_path)
        print(f"wrote lore index {index_path} ({count} chunks)", file=sys.stderr)
    except Exception:
        if conn is not None:
            conn.close()
        if tmp.exists():
            tmp.unlink()
        raise


def sqlite_record_from_row(row: tuple) -> LoreRecord:
    return LoreRecord(
        id=str(row[1]),
        path=str(row[2]),
        chunk=int(row[3]),
        anchor=str(row[4]),
        terms=[str(term) for term in json.loads(row[5])],
        source_date=str(row[6]),
        title=str(row[7]),
        text=str(row[8]),
    )


def sqlite_fetch_records(conn: sqlite3.Connection, rids: set[int]) -> list[LoreRecord]:
    if not rids:
        return []
    out: list[LoreRecord] = []
    ordered = sorted(rids)
    for start in range(0, len(ordered), SQLITE_MAX_VARS):
        chunk = ordered[start:start + SQLITE_MAX_VARS]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT rid, id, path, chunk, anchor, terms, source_date, title, text
            FROM records
            WHERE rid IN ({placeholders})
            """,
            chunk,
        )
        out.extend(sqlite_record_from_row(row) for row in rows)
    return out


def sqlite_fetch_record_by_path_chunk(conn: sqlite3.Connection, path: str, chunk: int) -> LoreRecord | None:
    row = conn.execute(
        """
        SELECT rid, id, path, chunk, anchor, terms, source_date, title, text
        FROM records
        WHERE path = ? AND chunk = ?
        """,
        (path, chunk),
    ).fetchone()
    return sqlite_record_from_row(row) if row else None


def escape_like_fragment(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def query_like_fragments(query: str) -> list[str]:
    fragments: list[str] = []
    for phrase in re.findall(r"`([^`]+)`", query):
        phrase = phrase.strip()
        if len(phrase) >= 3:
            fragments.append(phrase)
    for raw in TOKEN_RE.findall(query):
        if CJK_RE.match(raw):
            if len(raw) <= 8:
                fragments.append(raw)
            for size in (6, 5, 4, 3):
                if len(raw) >= size:
                    fragments.extend(raw[i:i + size] for i in range(0, len(raw) - size + 1))
        else:
            token = raw.lower()
            if len(token) >= 4 and token not in STOPWORDS:
                fragments.append(raw)

    out: list[str] = []
    seen: set[str] = set()
    for fragment in sorted(fragments, key=len, reverse=True):
        if fragment not in seen:
            out.append(fragment)
            seen.add(fragment)
        if len(out) >= 16:
            break
    return out


def sqlite_like_candidate_ids(conn: sqlite3.Connection, query: str, limit: int) -> set[int]:
    out: set[int] = set()
    for fragment in query_like_fragments(query):
        pattern = f"%{escape_like_fragment(fragment)}%"
        rows = conn.execute(
            """
            SELECT rid
            FROM records
            WHERE text LIKE ? ESCAPE '\\'
               OR title LIKE ? ESCAPE '\\'
               OR anchor LIKE ? ESCAPE '\\'
               OR path LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, limit),
        )
        out.update(int(row[0]) for row in rows)
        if len(out) >= limit:
            break
    return out


def sqlite_candidate_ids(
    conn: sqlite3.Connection,
    query: str,
    query_tokens: collections.Counter[str],
    top_k: int,
) -> set[int]:
    phrase_tokens: collections.Counter[str] = collections.Counter()
    for phrase in re.findall(r"`([^`]+)`", query):
        phrase_tokens.update(tokenize(phrase))

    base_tokens = phrase_tokens or query_tokens
    tokens = list(base_tokens)
    candidate_cap = max(top_k * 5000, 20000)
    if not tokens:
        return set()

    placeholders = ",".join("?" for _ in tokens)
    token_rows = conn.execute(
        f"""
        SELECT token, COUNT(*) AS doc_count
        FROM postings
        WHERE token IN ({placeholders})
        GROUP BY token
        ORDER BY doc_count ASC
        """,
        tokens,
    ).fetchall()
    if not token_rows:
        return sqlite_like_candidate_ids(conn, query, candidate_cap)

    min_candidates = max(top_k * 30, 80)
    token_postings: list[tuple[str, set[int]]] = []
    for token, _ in token_rows[:8]:
        rows = conn.execute("SELECT rid FROM postings WHERE token = ?", (token,))
        token_postings.append((str(token), {int(row[0]) for row in rows}))

    candidates = set(token_postings[0][1])
    for _, postings in token_postings[1:]:
        narrowed = candidates & postings
        if phrase_tokens:
            if narrowed:
                candidates = narrowed
        elif len(narrowed) >= min_candidates:
            candidates = narrowed

    if len(candidates) < max(top_k, 3) and len(token_postings) > 1:
        fallback: set[int] = set()
        for _, postings in token_postings:
            fallback.update(postings)
        if fallback:
            candidates = fallback

    if len(candidates) < max(top_k * 2, 6):
        candidates.update(sqlite_like_candidate_ids(conn, query, candidate_cap))

    if len(candidates) > candidate_cap:
        rare_token_candidates = sorted(candidates)[:candidate_cap]
        candidates = set(rare_token_candidates)
    return candidates


def expand_neighbor_records_sqlite(
    conn: sqlite3.Connection,
    selected: list[LoreRecord],
    after: str,
    before: str,
    neighbor_chunks: int,
) -> list[LoreRecord]:
    if neighbor_chunks <= 0:
        return selected
    out: list[LoreRecord] = []
    seen: set[tuple[str, int]] = set()
    selected_by_key = {(record.path, record.chunk): record for record in selected}
    for record in selected:
        for chunk in range(record.chunk - neighbor_chunks, record.chunk + neighbor_chunks + 1):
            key = (record.path, chunk)
            if key in seen:
                continue
            neighbor = selected_by_key.get(key)
            if neighbor is None:
                neighbor = sqlite_fetch_record_by_path_chunk(conn, record.path, chunk)
            if not neighbor or not date_in_range(neighbor.source_date, after, before):
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


def retrieve_records_sqlite(
    index_path: Path,
    query: str,
    top_k: int,
    min_score: float = 0.0,
    after: str = "",
    before: str = "",
    neighbor_chunks: int = 0,
    mmr_lambda: float = 0.9,
) -> list[LoreRecord]:
    query_tokens = expanded_query_tokens(query)
    conn = sqlite3.connect(index_path)
    try:
        candidate_ids = sqlite_candidate_ids(conn, query, query_tokens, top_k)
        scored: list[LoreRecord] = []
        for record in sqlite_fetch_records(conn, candidate_ids):
            if not date_in_range(record.source_date, after, before):
                continue
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
        return expand_neighbor_records_sqlite(conn, selected, after, before, neighbor_chunks)
    finally:
        conn.close()


def expanded_query_tokens(query: str) -> collections.Counter[str]:
    query_tokens = collections.Counter(tokenize(query))
    for token, count in list(query_tokens.items()):
        for expanded in QUERY_EXPANSIONS.get(token, []):
            query_tokens[expanded] += max(1, int(math.ceil(count * 0.5)))
    if not query_tokens:
        raise SystemExit("query produced no retrieval tokens")
    return query_tokens


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


def index_cmd(args: argparse.Namespace) -> None:
    pack = resolve_cli_path(args.pack)
    index_path = resolve_cli_path(args.out) if args.out else default_sqlite_index_path(pack)
    build_sqlite_index(pack, index_path, args.rebuild)


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
    pack = resolve_cli_path(args.pack)
    index_path: Path | None = None
    if not args.no_index:
        explicit_index = bool(args.index)
        index_path = resolve_cli_path(args.index) if explicit_index else default_sqlite_index_path(pack)
        should_use_index = (
            explicit_index
            or args.rebuild_index
            or index_path.exists()
            or pack.stat().st_size >= SQLITE_INDEX_AUTOBUILD_BYTES
        )
        if should_use_index:
            if args.rebuild_index or not sqlite_index_is_fresh(index_path, pack):
                build_sqlite_index(pack, index_path, args.rebuild_index)
            records = retrieve_records_sqlite(
                index_path,
                args.query,
                args.top_k,
                args.min_score,
                args.after,
                args.before,
                args.neighbor_chunks,
                args.mmr_lambda,
            )
            return cap_total(records, args.max_context_chars, args.excerpt_chars)

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


def archive_url_for_path(path: str, archive_base: str) -> str:
    filename = archive_filename_for_path(path)
    return f"{archive_base.rstrip('/')}/{urllib.parse.quote(filename)}"


def archive_filename_for_path(path: str) -> str:
    return Path(path).stem.lower().replace("：", "-")[:50]


def strip_html_text(value: str) -> str:
    value = HTML_BLOCK_RE.sub(" ", value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"</(p|div|section|article|li|blockquote|h[1-6]|tr|td|th)>", " ", value, flags=re.IGNORECASE)
    value = HTML_TAG_RE.sub(" ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def strip_prompt_source_text(value: str) -> str:
    value = re.sub(r"^\s{0,3}#{1,6}\s+.+?[:：]\s*$", " ", value, flags=re.MULTILINE)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_prompt_source_blocks(prompt: str) -> list[PromptSource]:
    sources: list[PromptSource] = []
    for match in PROMPT_SOURCE_RE.finditer(prompt):
        index = int(match.group(1))
        path = match.group(2).strip()
        chunk = int(match.group(3))
        title = (match.group(5) or Path(path).stem).strip()
        text = ""
        fenced = re.match(r"\n```text\n(.*?)\n```", prompt[match.end():], re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        sources.append(PromptSource(index=index, path=path, chunk=chunk, title=title, text=text))
    return sources


def default_sections_dump_path() -> Path | None:
    candidates = [
        Path.home() / "w" / "sayit-hono" / "scripts" / "sections-dump.json",
        Path.home() / "w" / "transcript" / "sections-dump.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_section_index_path() -> Path:
    return repo_root() / "dir-steering" / "out" / "archive-sections.sqlite"


def section_index_meta(sections_dump: Path) -> dict[str, str]:
    stat = sections_dump.stat()
    return {
        "format": SECTION_INDEX_FORMAT,
        "sections_dump": str(sections_dump.resolve()),
        "sections_dump_size": str(stat.st_size),
        "sections_dump_mtime_ns": str(stat.st_mtime_ns),
    }


def section_index_is_fresh(index_path: Path, sections_dump: Path) -> bool:
    if not index_path.exists():
        return False
    expected = section_index_meta(sections_dump)
    try:
        conn = sqlite3.connect(index_path)
        rows = dict(conn.execute("SELECT key, value FROM meta"))
        conn.close()
    except sqlite3.Error:
        return False
    return all(rows.get(key) == value for key, value in expected.items())


def build_section_index(sections_dump: Path, index_path: Path, rebuild: bool = False) -> None:
    if not rebuild and section_index_is_fresh(index_path, sections_dump):
        return

    print(f"building archive section index {index_path} from {sections_dump}", file=sys.stderr)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(f"{index_path.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    conn: sqlite3.Connection | None = None
    count = 0
    try:
        dump = json.loads(sections_dump.read_text(encoding="utf-8"))
        conn = sqlite3.connect(tmp)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.executescript("""
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE sections (
                filename TEXT NOT NULL,
                section_id INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                speaker TEXT NOT NULL,
                content TEXT NOT NULL
            );
        """)
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            sorted(section_index_meta(sections_dump).items()),
        )
        rows: list[tuple[str, int, str, str, str]] = []
        for filename, sections in dump.items():
            for section in sections:
                rows.append((
                    str(filename),
                    int(section["section_id"]),
                    str(section.get("display_name") or ""),
                    str(section.get("name") or ""),
                    strip_html_text(str(section.get("section_content") or "")),
                ))
                count += 1
                if len(rows) >= 1000:
                    conn.executemany(
                        "INSERT INTO sections(filename, section_id, display_name, speaker, content) VALUES (?, ?, ?, ?, ?)",
                        rows,
                    )
                    rows = []
        if rows:
            conn.executemany(
                "INSERT INTO sections(filename, section_id, display_name, speaker, content) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        conn.execute("CREATE INDEX sections_filename_idx ON sections(filename)")
        conn.commit()
        conn.close()
        conn = None
        tmp.replace(index_path)
        print(f"wrote archive section index {index_path} ({count} sections)", file=sys.stderr)
    except Exception:
        if conn is not None:
            conn.close()
        if tmp.exists():
            tmp.unlink()
        raise


def resolve_section_index(args: argparse.Namespace) -> Path | None:
    if args.no_section_anchors:
        return None
    index_path = resolve_cli_path(args.section_index) if args.section_index else default_section_index_path()
    dump_path: Path | None = resolve_cli_path(args.sections_dump) if args.sections_dump else default_sections_dump_path()
    if dump_path:
        if args.rebuild_section_index or not section_index_is_fresh(index_path, dump_path):
            build_section_index(dump_path, index_path, args.rebuild_section_index)
    return index_path if index_path.exists() else None


def score_section_match(source_text: str, section_text: str) -> float:
    source_plain = strip_prompt_source_text(source_text)
    section_plain = strip_html_text(section_text)
    if not source_plain or not section_plain:
        return 0.0
    if section_plain in source_plain:
        return 10000.0 + min(len(section_plain), 1000)
    probe = section_plain[: min(len(section_plain), 90)]
    if len(probe) >= 30 and probe in source_plain:
        return 5000.0 + len(probe)
    source_tokens = collections.Counter(tokenize(source_plain))
    section_tokens = collections.Counter(tokenize(section_plain))
    if not source_tokens or not section_tokens:
        return 0.0
    overlap = 0.0
    for token, count in section_tokens.items():
        if token in source_tokens:
            overlap += min(count, source_tokens[token])
    return overlap / math.sqrt(sum(section_tokens.values()) * sum(source_tokens.values()))


def find_best_section(index_path: Path, source: PromptSource, context: str = "") -> tuple[int, str] | None:
    filename = archive_filename_for_path(source.path)
    conn = sqlite3.connect(index_path)
    try:
        rows = conn.execute(
            "SELECT section_id, speaker, content FROM sections WHERE filename = ? ORDER BY section_id ASC",
            (filename,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    best: tuple[float, int, str] | None = None
    for section_id, speaker, content in rows:
        score = score_section_match(source.text, str(content))
        context_score = score_section_match(context, str(content)) if context else 0.0
        if context_score > 0:
            score = 100000.0 + context_score * 1000.0 + min(score, 5.0)
        if best is None or score > best[0]:
            best = (score, int(section_id), str(speaker or ""))
    if not best or best[0] <= 0:
        return None
    return best[1], best[2]


def resolve_prompt_source_sections(sources: list[PromptSource], index_path: Path | None) -> None:
    if index_path is None:
        return
    for source in sources:
        match = find_best_section(index_path, source)
        if match:
            source.section_id, source.section_speaker = match


def prompt_source_footnote(
    source: PromptSource,
    archive_base: str,
    section_index: Path | None = None,
    context: str = "",
) -> str:
    section_id = source.section_id
    section_speaker = source.section_speaker
    if section_index is not None:
        match = find_best_section(section_index, source, context)
        if match:
            section_id, section_speaker = match
    url = archive_url_for_path(source.path, archive_base)
    if section_id is not None:
        url = f"{url}#s{section_id}"
    label = source.title
    if section_speaker:
        label = f"{label} — {section_speaker}"
    return f"[{label}]({url})"


def parse_prompt_sources(prompt: str, archive_base: str, section_index: Path | None = None) -> dict[int, str]:
    sources = parse_prompt_source_blocks(prompt)
    resolve_prompt_source_sections(sources, section_index)
    return {source.index: prompt_source_footnote(source, archive_base) for source in sources}


def stream_markdown_footnotes(
    sources: dict[int, PromptSource],
    archive_base: str,
    all_sources: bool,
    section_index: Path | None = None,
) -> None:
    used: set[int] = set()
    contexts: dict[int, list[str]] = collections.defaultdict(list)
    history: collections.deque[str] = collections.deque(maxlen=360)
    state = "text"
    digits = ""

    def write(value: str) -> None:
        sys.stdout.write(value)
        sys.stdout.flush()

    while True:
        char = sys.stdin.read(1)
        if not char:
            break
        if state == "text":
            if char == "[":
                state = "citation"
                digits = ""
            else:
                write(char)
                history.append(char)
            continue

        if state == "citation":
            if char.isdigit() and len(digits) < 9:
                digits += char
                continue
            if char == "]" and digits:
                index = int(digits)
                used.add(index)
                contexts[index].append("".join(history))
                write(f"[^{index}]")
                for item in f"[^{index}]":
                    history.append(item)
                state = "text"
                digits = ""
                continue
            write("[" + digits + char)
            for item in "[" + digits + char:
                history.append(item)
            state = "text"
            digits = ""

    if state == "citation":
        write("[" + digits)
        for item in "[" + digits:
            history.append(item)

    footnote_indexes = sorted(sources) if all_sources else sorted(index for index in used if index in sources)
    if footnote_indexes:
        write("\n\n")
        for index in footnote_indexes:
            context = "\n".join(contexts.get(index, []))
            write(f"[^{index}]: {prompt_source_footnote(sources[index], archive_base, section_index, context)}\n")


def footnotes_cmd(args: argparse.Namespace) -> None:
    prompt_path = resolve_cli_path(args.prompt)
    prompt = prompt_path.read_text(encoding="utf-8")
    section_index = resolve_section_index(args)
    sources = {source.index: source for source in parse_prompt_source_blocks(prompt)}
    stream_markdown_footnotes(sources, args.archive_base, args.all, section_index)


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
    ap.add_argument("--index", default="",
                    help="SQLite lore index path; defaults to PACK.sqlite when present or for large packs")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the SQLite lore index before retrieving")
    ap.add_argument("--no-index", action="store_true",
                    help="disable the SQLite sidecar index and scan the JSONL pack in memory")


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

    index_ap = sub.add_parser("index")
    index_ap.add_argument("--pack", required=True)
    index_ap.add_argument("--out", default="", help="defaults to PACK.sqlite")
    index_ap.add_argument("--rebuild", action="store_true")
    index_ap.set_defaults(func=index_cmd)

    retrieve_ap = sub.add_parser("retrieve")
    add_retrieval_args(retrieve_ap)
    retrieve_ap.add_argument("--out", default="")
    retrieve_ap.set_defaults(func=retrieve_cmd)

    prompt_ap = sub.add_parser("prompt")
    add_retrieval_args(prompt_ap)
    prompt_ap.add_argument("--out", default="")
    prompt_ap.set_defaults(func=prompt_cmd)

    footnotes_ap = sub.add_parser("footnotes")
    footnotes_ap.add_argument("--prompt", required=True,
                              help="prompt file produced by the prompt subcommand")
    footnotes_ap.add_argument("--archive-base", default="https://archive.tw")
    footnotes_ap.add_argument("--sections-dump", default="",
                              help="sections-dump.json from sayit-hono; auto-detected from ~/w/sayit-hono")
    footnotes_ap.add_argument("--section-index", default="",
                              help="SQLite cache for resolving archive #s anchors")
    footnotes_ap.add_argument("--rebuild-section-index", action="store_true")
    footnotes_ap.add_argument("--no-section-anchors", action="store_true",
                              help="link to transcript pages without resolving section anchors")
    footnotes_ap.add_argument("--all", action="store_true",
                              help="append footnotes for every prompt source, not just cited sources")
    footnotes_ap.set_defaults(func=footnotes_cmd)

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
