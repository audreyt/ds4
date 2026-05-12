#!/usr/bin/env python3
"""Build a local calibration corpus for DeepSeek V4 Flash imatrix collection.

The quantizer needs activation statistics from prompts that look like the real
workload.  This script creates deterministic DS4-rendered chat prompts from the
repo itself, agent/tool conversations, bilingual tasks, and long-context code
reviews.  The output is intentionally plain JSONL/text so the imatrix collector
can consume it without depending on this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"

DEFAULT_SYSTEM = (
    "You are DeepSeek V4 Flash running locally. Answer accurately, preserve "
    "technical details, and use tools only when the prompt asks for tool use."
)

TOOLS_PROMPT = """## Tools

You have access to a set of tools to help answer the user question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as raw text and set `string="true"`. Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value.

### Available Tool Schemas

[
  {"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"},"description":{"type":"string"}},"required":["command"]}}},
  {"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"lines":{"type":"integer"}},"required":["path"]}}},
  {"type":"function","function":{"name":"edit","description":"Apply a small source edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}}},
  {"type":"function","function":{"name":"grep","description":"Search source files","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}}
]

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas."""

DSML_LIST_FILES = """
<｜DSML｜tool_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="description" string="true">list files in the project root</｜DSML｜parameter>
<｜DSML｜parameter name="command" string="true">find . -maxdepth 2 -type f | sort | head -200</｜DSML｜parameter>
<｜DSML｜parameter name="timeout" string="false">10</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

DSML_READ_FILE = """
<｜DSML｜tool_calls>
<｜DSML｜invoke name="read">
<｜DSML｜parameter name="path" string="true">ds4.c</｜DSML｜parameter>
<｜DSML｜parameter name="start" string="false">14000</｜DSML｜parameter>
<｜DSML｜parameter name="lines" string="false">120</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""


@dataclass
class Record:
    rid: str
    category: str
    mode: str
    source: str
    messages: list[dict[str, str]]
    rendered: str


def escape_tool_result(text: str) -> str:
    """Match the renderer's intent: tool outputs must not become live DSML."""

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render(messages: list[dict[str, str]], mode: str, tools: bool = False) -> str:
    """Render the subset of DS4 chat syntax needed for calibration.

    This mirrors ds4_server.c's prompt shape: BOS, concatenated system text,
    user/tool/assistant turns, then an assistant prefix using <think> or
    </think>.  Existing assistant answers close with EOS.
    """

    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    if tools:
        system_parts.append(TOOLS_PROMPT)
    system = "\n\n".join(p for p in system_parts if p)
    think = mode == "think"

    out = [BOS, system]
    pending_assistant = False
    pending_tool_result = False
    last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if role == "user":
            out.extend([USER, content])
            pending_assistant = True
            pending_tool_result = False
        elif role in ("tool", "function"):
            if not pending_tool_result:
                out.append(USER)
            out.extend(["<tool_result>", escape_tool_result(content), "</tool_result>"])
            pending_assistant = True
            pending_tool_result = True
        elif role == "assistant":
            if pending_assistant:
                out.append(ASSISTANT)
                if think and i > last_user_idx:
                    out.extend(["<think>", msg.get("reasoning", ""), "</think>"])
                else:
                    out.append("</think>")
            out.append(content)
            if msg.get("dsml"):
                out.append(msg["dsml"])
            out.append(EOS)
            pending_assistant = False
            pending_tool_result = False

    if pending_assistant:
        out.extend([ASSISTANT, "<think>" if think else "</think>"])

    return "".join(out)


def stable_id(category: str, source: str, mode: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(category.encode())
    h.update(b"\0")
    h.update(source.encode())
    h.update(b"\0")
    h.update(mode.encode())
    h.update(b"\0")
    h.update(text[:4096].encode("utf-8", "ignore"))
    return f"{category}-{h.hexdigest()[:12]}"


def add_record(records: list[Record], category: str, source: str,
               messages: list[dict[str, str]], *, tools: bool = False,
               modes: Iterable[str] = ("nothink", "think")) -> None:
    for mode in modes:
        rendered = render(messages, mode, tools=tools)
        rid = stable_id(category, source, mode, rendered)
        records.append(Record(rid, category, mode, source, messages, rendered))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            nl = text.rfind("\n", i + size // 2, j)
            if nl > i:
                j = nl
        part = text[i:j].strip()
        if part:
            chunks.append(part)
        if j == len(text):
            break
        i = max(j - overlap, i + 1)
    return chunks


def code_prompt(path: str, chunk: str, task: str) -> list[dict[str, str]]:
    lang = "metal" if path.endswith(".metal") else "c"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": f"{task}\n\nFile: {path}\n\n```{lang}\n{chunk}\n```"},
    ]


def doc_prompt(path: str, chunk: str, task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": f"{task}\n\nDocument: {path}\n\n{chunk}"},
    ]


def make_source_records(root: Path, records: list[Record]) -> None:
    files = [
        "ds4.c", "ds4_server.c", "ds4_cli.c", "ds4_metal.m", "ds4.h", "ds4_gpu.h",
        "README.md", "AGENT.md", "gguf-tools/README.md",
        "gguf-tools/imatrix/README.md", "gguf-tools/imatrix/dataset/README.md",
        "gguf-tools/quality-testing/README.md",
    ]
    files += [str(p.relative_to(root)) for p in sorted((root / "metal").glob("*.metal"))]
    tasks = [
        "Review this excerpt for correctness risks, memory lifetime issues, and performance bottlenecks.",
        "Explain what this code does and identify the inference stage it belongs to.",
        "Suggest a minimal correctness-preserving optimization for this excerpt.",
        "Trova bug sottili e spiega quali invarianti dell'inferenza possono rompersi.",
    ]

    for name in files:
        text = read_text(root / name)
        if not text:
            continue
        is_code = name.endswith((".c", ".h", ".m", ".metal"))
        size = 2600 if is_code else 3200
        chunks = chunk_text(text, size=size, overlap=180)
        for idx, chunk in enumerate(chunks):
            task = tasks[(idx + len(name)) % len(tasks)]
            msgs = code_prompt(name, chunk, task) if is_code else doc_prompt(name, chunk, task)
            add_record(records, "source", f"{name}:{idx}", msgs)


def make_agent_records(records: list[Record]) -> None:
    scenarios = [
        (
            "opencode-list-files",
            [
                {"role": "system", "content": "You are a coding agent operating in a local repository. Be terse and use tools for filesystem inspection."},
                {"role": "user", "content": "list files in the current dir"},
                {"role": "assistant", "content": "", "dsml": DSML_LIST_FILES},
                {"role": "tool", "content": ".git\nREADME.md\nds4.c\nds4_server.c\nmetal/\nmisc/\ntests/\n"},
                {"role": "user", "content": "What files matter for the CLI?"},
            ],
            True,
        ),
        (
            "opencode-read-source",
            [
                {"role": "system", "content": "You are a coding agent. Inspect before editing."},
                {"role": "user", "content": "Find where DS4 renders the chat template and explain how tool calls are preserved."},
                {"role": "assistant", "content": "", "dsml": DSML_READ_FILE},
                {"role": "tool", "content": "static char *render_chat_prompt_text(...) { /* abbreviated renderer output */ }\n"},
                {"role": "user", "content": "Now summarize the invariants that must not drift."},
            ],
            True,
        ),
        (
            "kv-cache-debug",
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": "A server request has prompt=16358 cached=11412 suffix=4946. Explain why suffix prefill must be chunked instead of decoded token by token."},
                {"role": "assistant", "content": "The suffix is long enough that single-token decode would waste the model's batched prefill path. The implementation should start at the cached prefix and process the remaining tokens in 2048-token chunks."},
                {"role": "user", "content": "Now write a checklist for testing this with context cache hits."},
            ],
            False,
        ),
        (
            "metal-performance-debug",
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": "The prefill chunk log starts at 230 tok/s and drops toward 150 tok/s at 16k context. Rank likely causes in DS4 Flash attention."},
                {"role": "assistant", "content": "The slope usually comes from indexed compressed attention, raw window attention, compressor state, and common MoE/HC cost. Measure per-layer ratio-4 and ratio-128 stages separately."},
                {"role": "user", "content": "Which measurements would distinguish score-kernel slope from attention-scan slope?"},
            ],
            False,
        ),
    ]
    for name, msgs, tools in scenarios:
        add_record(records, "agent", name, msgs, tools=tools)


def make_general_records(records: list[Record]) -> None:
    english = [
        "Explain how a B-tree insertion works, including splits and the root special case.",
        "Write a concise design for a TCP echo server that handles slow clients without blocking other clients.",
        "Compare mmap-backed model loading with copying all weights into private buffers on macOS.",
        "Derive why RMSNorm can be implemented with one sum of squares and one scale pass.",
        "Explain indexed sparse attention in a mixture-of-experts language model.",
        "Create a careful migration plan from a C++ graph executor to a pure C model-specific executor.",
        "Find the bug in a ring buffer where speculative writes may overwrite still-visible rows.",
        "Explain why quantization imatrices should be collected from realistic prompts instead of random text.",
        "Summarize the difference between prefill and decode in transformer inference.",
        "Write a troubleshooting guide for an OpenAI-compatible local chat server that hangs during tool calls.",
    ]
    italian = [
        "Spiega come funziona l'inserimento in un B-tree, inclusi split e caso della radice.",
        "Scrivi un progetto conciso per un server TCP echo che gestisce client lenti senza bloccare gli altri.",
        "Confronta il caricamento mmap dei pesi con la copia completa in buffer privati su macOS.",
        "Deriva perche RMSNorm richiede una somma dei quadrati e un passaggio di scala.",
        "Spiega l'attenzione sparsa indicizzata in un modello linguistico mixture-of-experts.",
        "Crea un piano di migrazione da un executor a grafo C++ a un executor specifico in C puro.",
        "Trova il bug in un ring buffer dove scritture speculative possono sovrascrivere righe ancora visibili.",
        "Spiega perche le imatrix di quantizzazione vanno raccolte da prompt realistici e non testo casuale.",
        "Riassumi la differenza tra prefill e decode nell'inferenza transformer.",
        "Scrivi una guida di debug per un server chat locale compatibile OpenAI che si blocca nei tool call.",
    ]
    for idx, prompt in enumerate(english + italian):
        add_record(records, "general", f"general:{idx}", [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ])


def make_long_context_records(root: Path, records: list[Record]) -> None:
    sources = [
        ("README.md", read_text(root / "README.md")),
        ("AGENT.md", read_text(root / "AGENT.md")),
        ("METAL.md", read_text(root / "METAL.md")),
        ("ds4_server.c", read_text(root / "ds4_server.c")),
        ("ds4_metal.m", read_text(root / "ds4_metal.m")),
        ("metal/dsv4_hc.metal", read_text(root / "metal/dsv4_hc.metal")),
        ("metal/moe.metal", read_text(root / "metal/moe.metal")),
    ]
    blocks = []
    for name, text in sources:
        for i, chunk in enumerate(chunk_text(text, size=5200, overlap=120)[:6]):
            blocks.append(f"### {name} chunk {i}\n{chunk}")
    random.Random(7).shuffle(blocks)
    for i in range(0, len(blocks), 4):
        group = blocks[i:i + 4]
        if len(group) < 2:
            continue
        body = "\n\n".join(group)
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": (
                "Read the following repository excerpts as one long context. "
                "Identify the three most important invariants for correctness and "
                "the two most likely performance bottlenecks.\n\n" + body
            )},
        ]
        add_record(records, "long_context", f"long:{i//4}", msgs)


def write_outputs(outdir: Path, records: list[Record]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: (r.category, r.source, r.mode, r.rid))

    jsonl = outdir / "prompts.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r.rid,
                "category": r.category,
                "mode": r.mode,
                "source": r.source,
                "messages": r.messages,
                "rendered": r.rendered,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write_rendered(path: Path, rows: Iterable[Record]) -> int:
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for count, r in enumerate(rows, start=1):
                f.write(f"\n\n===== DS4_IMATRIX_PROMPT {r.rid} {r.category} {r.mode} {r.source} =====\n")
                f.write(r.rendered)
        return count

    write_rendered(outdir / "rendered_prompts.txt", records)
    write_rendered(outdir / "rendered_prompts_nothink.txt", [r for r in records if r.mode == "nothink"])
    write_rendered(outdir / "rendered_prompts_think.txt", [r for r in records if r.mode == "think"])

    categories: dict[str, int] = {}
    modes: dict[str, int] = {}
    bytes_by_category: dict[str, int] = {}
    for r in records:
        categories[r.category] = categories.get(r.category, 0) + 1
        modes[r.mode] = modes.get(r.mode, 0) + 1
        bytes_by_category[r.category] = bytes_by_category.get(r.category, 0) + len(r.rendered.encode("utf-8"))

    rendered_bytes = sum(len(r.rendered.encode("utf-8")) for r in records)
    manifest = {
        "version": 1,
        "purpose": "DeepSeek V4 Flash imatrix calibration prompts",
        "record_count": len(records),
        "rendered_utf8_bytes": rendered_bytes,
        "rough_token_estimate_bytes_div_4": rendered_bytes // 4,
        "categories": categories,
        "modes": modes,
        "bytes_by_category": bytes_by_category,
        "files": {
            "jsonl": "prompts.jsonl",
            "all_rendered": "rendered_prompts.txt",
            "nothink_rendered": "rendered_prompts_nothink.txt",
            "think_rendered": "rendered_prompts_think.txt",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output directory. Defaults to this script's directory.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = find_repo_root(script_dir)
    outdir = Path(args.out).resolve() if args.out else script_dir

    records: list[Record] = []
    make_source_records(root, records)
    make_agent_records(records)
    make_general_records(records)
    make_long_context_records(root, records)
    write_outputs(outdir, records)

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    print(f"wrote {manifest['record_count']} prompts to {outdir}")
    print(f"rendered bytes: {manifest['rendered_utf8_bytes']}")
    print(f"rough tokens: {manifest['rough_token_estimate_bytes_div_4']}")


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "ds4.c").exists() and (path / "metal").is_dir():
            return path
    raise RuntimeError(f"could not find ds4.c repository root from {start}")


if __name__ == "__main__":
    main()
