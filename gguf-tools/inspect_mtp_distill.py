#!/usr/bin/env python3
"""Inspect a ds4 MTP distillation dump written by --mtp-distill-out."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


HEADER = struct.Struct("<8s8I3Q")
PREFIX = struct.Struct("<IiiIffII")
TOP = struct.Struct("<iff")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--records", type=int, default=5)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.path)
    with path.open("rb") as fp:
        header = fp.read(HEADER.size)
        if len(header) != HEADER.size:
            raise SystemExit(f"{path}: short header")
        (
            magic,
            version,
            header_bytes,
            n_embd,
            n_hc,
            vocab,
            top_k,
            prompt_tokens,
            ctx_size,
            record_bytes,
            records,
            reserved,
        ) = HEADER.unpack(header)
        if magic != b"DS4MTPD1" or version != 1 or header_bytes != HEADER.size:
            raise SystemExit(f"{path}: unsupported MTP distill dump")
        if reserved != 0:
            print(f"reserved={reserved}")

        print(f"path={path}")
        print(f"records={records} top_k={top_k} record_bytes={record_bytes}")
        print(f"shape: n_hc={n_hc} n_embd={n_embd} vocab={vocab} prompt_tokens={prompt_tokens} ctx={ctx_size}")

        hc_bytes = n_hc * n_embd * 4
        want_records = min(args.records, records)
        want_top = min(args.top, top_k)
        for i in range(want_records):
            prefix = fp.read(PREFIX.size)
            if len(prefix) != PREFIX.size:
                raise SystemExit(f"{path}: truncated record {i}")
            pos, token, target, n_scores, target_logit, target_logprob, flags, _ = PREFIX.unpack(prefix)
            fp.seek(hc_bytes, 1)
            top = []
            for j in range(top_k):
                item = fp.read(TOP.size)
                if len(item) != TOP.size:
                    raise SystemExit(f"{path}: truncated top-k record {i}")
                if j < want_top:
                    top.append(TOP.unpack(item))
            print(
                f"record {i}: pos={pos} token={token} target={target} "
                f"logit={target_logit:.5g} logprob={target_logprob:.5g} scores={n_scores} flags={flags}"
            )
            print("  top:", ", ".join(f"{tid}:{logit:.4g}/{logprob:.4g}" for tid, logit, logprob in top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
