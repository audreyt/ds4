#!/usr/bin/env python3
"""Build deterministic long-lore retrieval/evaluation cases.

The cases are "needle" questions: each query quotes a short passage from one
dated source, while the expected answer asks for the source date/title.  This
keeps compose-only evaluation honest, because the scored source fields are not
present in the query unless retrieval actually found the right chunk.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import lore_cag


SPEAKER_RE = re.compile(r"^\s{0,3}(#{2,6}\s*)?[^:\n]{1,80}[:：]\s*$")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")


def clean_text(value: str) -> str:
    value = MARKDOWN_LINK_RE.sub(r"\1", value)
    value = CODE_SPAN_RE.sub(r"\1", value)
    value = re.sub(r"[*_~#>\[\]()`]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def select_evenly(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[len(paths) // 2]]
    selected: list[Path] = []
    seen: set[Path] = set()
    for i in range(limit):
        idx = round(i * (len(paths) - 1) / (limit - 1))
        path = paths[idx]
        if path in seen:
            continue
        selected.append(path)
        seen.add(path)
    return selected


def candidate_passages(text: str, max_needle_chars: int) -> list[str]:
    passages: list[str] = []
    for raw in re.split(r"\n\s*\n|(?<=[.!?。！？])\s+", text):
        line = clean_text(raw)
        if not line:
            continue
        if line.startswith("http") or "://" in line:
            continue
        if SPEAKER_RE.match(line):
            continue
        if len(line) < 45 and not lore_cag.CJK_RE.search(line):
            continue
        if len(line) < 24:
            continue
        if len(line) > max_needle_chars:
            line = line[:max_needle_chars].rsplit(" ", 1)[0].strip() or line[:max_needle_chars].strip()
        tokens = set(lore_cag.tokenize(line))
        if len(tokens) < 4:
            continue
        passages.append(line)
    return passages


def build_document_frequencies(paths: list[Path], max_file_chars: int) -> collections.Counter[str]:
    df: collections.Counter[str] = collections.Counter()
    for path in paths:
        text = lore_cag.read_text(path, max_file_chars)
        df.update(set(lore_cag.tokenize(text)))
    return df


def choose_needle(
    path: Path,
    text: str,
    df: collections.Counter[str],
    doc_count: int,
    max_needle_chars: int,
    source_lowers: list[str],
) -> str:
    title_tokens = set(lore_cag.tokenize(lore_cag.source_title(path, text)))
    path_tokens = set(lore_cag.tokenize(path.name))
    scored: list[tuple[float, str]] = []
    for passage in candidate_passages(text, max_needle_chars):
        tokens = [token for token in lore_cag.tokenize(passage) if token not in title_tokens and token not in path_tokens]
        if len(set(tokens)) < 3:
            continue
        rarity = sum(math.log((doc_count + 1.0) / (df.get(token, 0) + 1.0)) + 1.0 for token in set(tokens))
        length_bonus = min(len(passage), max_needle_chars) / max_needle_chars
        cjk_bonus = 0.2 if lore_cag.CJK_RE.search(passage) else 0.0
        score = rarity + length_bonus + cjk_bonus
        scored.append((score, passage))
    for _, passage in sorted(scored, reverse=True)[:80]:
        passage_lower = passage.lower()
        if sum(1 for source in source_lowers if passage_lower in source) == 1:
            return passage
    raise ValueError(f"{path}: could not extract a source-unique needle passage")


def make_case(path: Path, text: str, needle: str, index: int) -> dict:
    date = lore_cag.source_date(path)
    title = lore_cag.source_title(path, text)
    title_aliases = [path.name, title]
    if date and title.startswith(f"{date} "):
        title_aliases.append(title[len(date):].strip(" -:："))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()[:80]
    return {
        "id": f"long_lore_{index:03d}_{slug}",
        "query": (
            "Which dated transcript in the Audrey long-lore corpus contains this passage: "
            f"`{needle}`? Return the source date and source title, and cite the lore excerpt."
        ),
        "expect": [date],
        "expect_any": [title_aliases],
        "forbid": [
            "I do not see",
            "not in the provided lore",
            "no transcript",
            "cannot determine",
        ],
        "source": str(path),
        "source_date": date,
        "source_title": title,
        "needle": needle,
        "tags": ["long-lore", "needle", "cjk" if lore_cag.CJK_RE.search(needle) else "latin"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create long-lore CAG evaluation cases from dated markdown sources.")
    ap.add_argument("--doc", action="append", required=True, help="source file or directory; may be repeated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--after", default="")
    ap.add_argument("--before", default="")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-file-chars", type=int, default=240_000)
    ap.add_argument("--max-needle-chars", type=int, default=180)
    args = ap.parse_args()

    root = lore_cag.repo_root()
    paths = lore_cag.iter_paths(args.doc, root, {".md"}, set(lore_cag.DEFAULT_EXCLUDES))
    paths = lore_cag.filter_paths_by_date(paths, args.after, args.before)
    paths.sort(key=lambda item: (lore_cag.source_date(item), item.name))
    source_texts = [lore_cag.read_text(path, args.max_file_chars) for path in paths]
    source_lowers = [text.lower() for text in source_texts]
    df: collections.Counter[str] = collections.Counter()
    for text in source_texts:
        df.update(set(lore_cag.tokenize(text)))
    selected = select_evenly(paths, args.limit)

    cases: list[dict] = []
    skipped: list[str] = []
    for path in selected:
        text = source_texts[paths.index(path)]
        try:
            needle = choose_needle(path, text, df, len(paths), args.max_needle_chars, source_lowers)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        cases.append(make_case(path, text, needle, len(cases) + 1))

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ds4-lore-testset-v1",
        "source_count": len(paths),
        "case_count": len(cases),
        "after": args.after,
        "before": args.before,
        "cases": cases,
        "skipped": skipped,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"built {len(cases)} cases from {len(paths)} dated sources")
    if skipped:
        print(f"skipped {len(skipped)} sources without usable needles")


if __name__ == "__main__":
    main()
