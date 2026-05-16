#!/usr/bin/env python3
"""Build DS4 steering prompts from documents, then optionally capture a vector.

This is the cheap "doc-to-steering" path:

    documents -> paired target/contrast probes -> build_direction.py -> .f32

The generated target probes include short reference excerpts and ask the model
to answer like a DS4 maintainer.  The contrast probes ask for the same kind of
answer in generic terms, without project-local details.  The resulting direction
is useful as a document-conditioned behavior/register nudge; it is not a memory
store and should be paired with normal context/RAG when exact facts matter.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTS = {
    ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".inc", ".m", ".metal",
    ".md", ".py", ".sh", ".txt", ".json", ".jsonl",
}

DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    "dir-steering/out",
    "gguf",
}

STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "args",
    "because", "before", "being", "bool", "but", "can", "char", "const",
    "default", "does", "each", "else", "false", "file", "float", "for",
    "from", "have", "help", "here", "into", "int", "long", "make", "may",
    "more", "must", "not", "one", "only", "other", "path", "read", "return",
    "same", "should", "static", "str", "struct", "that", "the", "then",
    "there", "this", "true", "uint32_t", "uint64_t", "use", "used", "using",
    "void", "when", "where", "with", "would", "write",
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(")
SPACE_RE = re.compile(r"\s+")


@dataclass
class DocChunk:
    path: Path
    rel_path: str
    index: int
    text: str
    anchor: str
    terms: list[str]
    score: float


def parse_csv_set(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return set(default)
    out: set[str] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
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


def iter_doc_paths(inputs: list[Path], root: Path, exts: set[str], excludes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"{raw}: does not exist")
        if path.is_file():
            if path.suffix in exts and not is_excluded(path, root, excludes):
                paths.append(path)
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix in exts and not is_excluded(candidate, root, excludes):
                paths.append(candidate.resolve())
    deduped = sorted(dict.fromkeys(paths))
    if not deduped:
        raise SystemExit("no document files found")
    return deduped


def read_text(path: Path, max_chars: int) -> str:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="ignore")
    text = text.replace("\x00", " ")
    if max_chars > 0 and len(text) > max_chars:
        # Preserve both file overview and later implementation details.
        head = max_chars // 2
        tail = max_chars - head
        text = text[:head] + "\n\n[... middle omitted by build_doc_direction.py ...]\n\n" + text[-tail:]
    return text


def flatten(text: str, limit: int) -> str:
    text = SPACE_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{cut} ..."


def split_chunks(text: str, max_chars: int, min_chars: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
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

    for line in lines:
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


def term_counts(text: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for token in IDENT_RE.findall(text):
        lower = token.lower()
        if lower in STOPWORDS or len(lower) < 3 or lower.isdigit():
            continue
        weight = 1
        if "_" in token:
            weight += 3
        if token.startswith(("ds4", "DS4", "dsv4", "Dsv4")):
            weight += 4
        if token.isupper() and len(token) > 3:
            weight += 1
        counts[token] += weight
    return counts


def choose_anchor(path: Path, text: str, terms: list[str]) -> str:
    headings = [flatten(h, 80) for h in HEADING_RE.findall(text)]
    if headings:
        return headings[0]
    funcs = [name for name in FUNC_RE.findall(text) if name.lower() not in STOPWORDS]
    if funcs:
        return funcs[0]
    if terms:
        return ", ".join(terms[:3])
    return path.stem


def score_chunk(text: str, terms: list[str]) -> float:
    score = min(len(text), 4000) / 1000.0
    score += sum(3.0 if "_" in term else 1.0 for term in terms[:8])
    if HEADING_RE.search(text):
        score += 5.0
    if "DS4" in text or "DeepSeek" in text or "dir-steering" in text:
        score += 3.0
    return score


def build_chunks(
    paths: list[Path],
    root: Path,
    max_file_chars: int,
    max_chunk_chars: int,
    min_chunk_chars: int,
    max_per_file: int,
) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for path in paths:
        text = read_text(path, max_file_chars)
        file_chunks: list[DocChunk] = []
        rel_path = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        for index, chunk_text in enumerate(split_chunks(text, max_chunk_chars, min_chunk_chars), 1):
            counts = term_counts(chunk_text)
            terms = [term for term, _ in counts.most_common(12)]
            anchor = choose_anchor(path, chunk_text, terms)
            file_chunks.append(DocChunk(
                path=path,
                rel_path=rel_path,
                index=index,
                text=chunk_text,
                anchor=anchor,
                terms=terms,
                score=score_chunk(chunk_text, terms),
            ))
        file_chunks.sort(key=lambda item: item.score, reverse=True)
        chunks.extend(file_chunks[:max_per_file])
    chunks.sort(key=lambda item: item.score, reverse=True)
    return chunks


def select_balanced(chunks: list[DocChunk], limit: int) -> list[DocChunk]:
    """Pick high-scoring chunks while keeping large files from monopolizing."""
    by_path: dict[str, list[DocChunk]] = collections.defaultdict(list)
    for chunk in chunks:
        by_path[chunk.rel_path].append(chunk)
    for items in by_path.values():
        items.sort(key=lambda item: item.score, reverse=True)

    order = sorted(by_path, key=lambda rel: by_path[rel][0].score, reverse=True)
    selected: list[DocChunk] = []
    while len(selected) < limit:
        progressed = False
        for rel in order:
            if by_path[rel]:
                selected.append(by_path[rel].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


TASK_TEMPLATES = [
    "Explain what matters about {anchor} when changing or debugging this codebase.",
    "Review a proposed change around {anchor}; call out correctness and integration risks.",
    "Summarize the DS4-specific behavior around {anchor} for a new contributor.",
    "A regression appears near {anchor}. Describe the investigation path and likely invariants.",
]


def genericize_task(task: str, anchor: str) -> str:
    return task.replace(anchor, "this component")


def make_prompt_pair(chunk: DocChunk, prompt_chars: int, index: int) -> tuple[str, str]:
    terms = ", ".join(chunk.terms[:8]) if chunk.terms else chunk.anchor
    task = TASK_TEMPLATES[index % len(TASK_TEMPLATES)].format(anchor=chunk.anchor)
    generic_task = genericize_task(task, chunk.anchor)
    excerpt = flatten(chunk.text, prompt_chars)

    good = (
        f"Reference excerpt from {chunk.rel_path}: <doc> {excerpt} </doc> "
        f"Answer as a DS4 maintainer. Use the excerpt's project-specific names, "
        f"constraints, implementation caveats, and terminology. Key cues: {terms}. "
        f"Task: {task}"
    )
    bad = (
        "Answer as a general software or ML systems explanation with no access to "
        "the DS4 repository. Avoid project-specific identifiers, source-local facts, "
        f"and maintainer conventions. Task: {generic_task}"
    )
    return flatten(good, prompt_chars + 900), flatten(bad, 900)


def write_prompt_file(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def sidecar_paths(out: Path, prompt_dir: Path | None) -> tuple[Path, Path, Path]:
    root = prompt_dir or out.parent
    stem = out.stem
    return (
        root / f"{stem}.doc-good.txt",
        root / f"{stem}.doc-bad.txt",
        root / f"{stem}.doc-to-steering.json",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description="Generate doc-conditioned DS4 steering prompts.")
    ap.add_argument("--doc", action="append", required=True,
                    help="document file or directory; may be repeated")
    ap.add_argument("--out", default="dir-steering/out/doc_direction.json",
                    help="final build_direction.py output JSON path")
    ap.add_argument("--prompt-dir", default="",
                    help="directory for generated prompt sidecars; defaults beside --out")
    ap.add_argument("--include-ext", default="",
                    help="comma-separated extensions; default is common DS4 source/doc files")
    ap.add_argument("--exclude", action="append", default=[],
                    help="path fragment to skip; may be repeated")
    ap.add_argument("--max-prompts", type=int, default=64)
    ap.add_argument("--max-prompts-per-file", type=int, default=10)
    ap.add_argument("--max-file-chars", type=int, default=200_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2400)
    ap.add_argument("--min-chunk-chars", type=int, default=300)
    ap.add_argument("--prompt-chars", type=int, default=1900)
    ap.add_argument("--build", action="store_true",
                    help="run build_direction.py after writing prompt files")

    # Pass-through options for build_direction.py.
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument("--ctx", type=int, default=768)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="ffn_out", choices=("ffn_out", "attn_out"))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--pair-normalize", action="store_true")
    ap.add_argument("--no-orthogonalize", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out
    prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
    if prompt_dir and not prompt_dir.is_absolute():
        prompt_dir = repo_root / prompt_dir
    good_path, bad_path, recipe_path = sidecar_paths(out, prompt_dir)

    exts = parse_csv_set(args.include_ext, DEFAULT_EXTS)
    excludes = set(DEFAULT_EXCLUDES)
    excludes.update(args.exclude)
    docs = iter_doc_paths([Path(item) for item in args.doc], repo_root, exts, excludes)
    chunks = select_balanced(build_chunks(
        docs,
        repo_root,
        args.max_file_chars,
        args.max_chunk_chars,
        args.min_chunk_chars,
        args.max_prompts_per_file,
    ), args.max_prompts)
    if not chunks:
        raise SystemExit("no usable document chunks found")

    good_prompts: list[str] = []
    bad_prompts: list[str] = []
    for i, chunk in enumerate(chunks):
        good, bad = make_prompt_pair(chunk, args.prompt_chars, i)
        good_prompts.append(good)
        bad_prompts.append(bad)

    write_prompt_file(good_path, good_prompts)
    write_prompt_file(bad_path, bad_prompts)

    build_cmd = [
        sys.executable,
        str(Path(__file__).with_name("build_direction.py")),
        "--ds4", args.ds4,
        "--model", args.model,
        "--good-file", str(good_path),
        "--bad-file", str(bad_path),
        "--out", str(out),
        "--ctx", str(args.ctx),
        "--system", args.system,
        "--component", args.component,
    ]
    if args.think:
        build_cmd.append("--think")
    if args.pair_normalize:
        build_cmd.append("--pair-normalize")
    if args.no_orthogonalize:
        build_cmd.append("--no-orthogonalize")

    recipe = {
        "format": "ds4-doc-to-steering-prompts-v1",
        "prompt_count": len(good_prompts),
        "good_file": str(good_path),
        "bad_file": str(bad_path),
        "out": str(out),
        "component": args.component,
        "ctx": args.ctx,
        "thinking": bool(args.think),
        "docs": [str(path) for path in docs],
        "chunks": [
            {
                "path": chunk.rel_path,
                "index": chunk.index,
                "anchor": chunk.anchor,
                "terms": chunk.terms[:8],
                "score": round(chunk.score, 3),
            }
            for chunk in chunks
        ],
        "build_command": build_cmd,
        "note": (
            "Negative runtime FFN scales amplify the document-conditioned target "
            "register; positive scales suppress it. Pair with context/RAG for exact facts."
        ),
    }
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")

    print(f"wrote {good_path}")
    print(f"wrote {bad_path}")
    print(f"wrote {recipe_path}")
    print(f"selected {len(chunks)} chunks from {len(docs)} files")

    if args.build:
        print("running build_direction.py", flush=True)
        subprocess.run(build_cmd, cwd=repo_root, check=True)
    else:
        print("dry run only; add --build to capture activations and write the .f32 direction")


if __name__ == "__main__":
    main()
