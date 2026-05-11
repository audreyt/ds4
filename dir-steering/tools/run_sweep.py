#!/usr/bin/env python3
"""Run a small steering scale sweep through ds4.

This is intentionally thin: it exercises the same public CLI options users
will use in production and leaves all inference behavior inside ds4.
"""

import argparse
import subprocess
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="ds4flash.gguf")
    ap.add_argument("--direction", required=True,
                    help="flat f32 vector file produced by build_direction.py")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--attn-scale", type=float, default=0.0)
    ap.add_argument("--nothink", action="store_true")
    args = ap.parse_args()

    prompts = read_prompts(Path(args.prompts))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        for scale in scales:
            print("-" * 80)
            print(f"FFN scale: {scale:g}")
            cmd = [
                args.ds4,
                "-m", args.model,
                "--ctx", str(args.ctx),
                "-n", str(args.tokens),
                "--temp", "0",
                "--dir-steering-file", args.direction,
                "--dir-steering-ffn", str(scale),
                "--dir-steering-attn", str(args.attn_scale),
                "-p", prompt,
            ]
            if args.nothink:
                cmd.append("--nothink")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
