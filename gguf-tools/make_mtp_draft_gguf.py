#!/usr/bin/env python3
"""Build a bootstrapped DS4 MTP-drafter GGUF from an existing target GGUF.

This does not train an MTP module.  It creates a valid MTP-only support GGUF
for ds4 by copying one target transformer block into the `mtp.0.*` tensor names
and synthesizing the MTP input projectors.  The resulting file is useful as a
local seed artifact and a sanity target for later distillation.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO, Callable


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

TENSOR_F32 = 0
TENSOR_F16 = 1
TENSOR_Q8_0 = 8
TENSOR_Q2_K = 10
TENSOR_Q4_K = 12
TENSOR_IQ2_XXS = 16
TENSOR_I32 = 26

DS4_N_EMBD = 4096
DS4_N_HC = 4
DS4_N_EXPERT = 256

TYPE_BLOCKS = {
    TENSOR_F32: (1, 4),
    TENSOR_F16: (1, 2),
    TENSOR_Q8_0: (32, 34),
    TENSOR_Q2_K: (256, 84),
    TENSOR_Q4_K: (256, 144),
    TENSOR_IQ2_XXS: (256, 66),
    TENSOR_I32: (1, 4),
}

TYPE_NAMES = {
    TENSOR_F32: "f32",
    TENSOR_F16: "f16",
    TENSOR_Q8_0: "q8_0",
    TENSOR_Q2_K: "q2_k",
    TENSOR_Q4_K: "q4_k",
    TENSOR_IQ2_XXS: "iq2_xxs",
    TENSOR_I32: "i32",
}


@dataclass
class TensorMeta:
    name: str
    dims: list[int]
    type: int
    old_offset: int
    size: int
    new_offset: int = 0


@dataclass
class KvRecord:
    key: str
    type: int
    raw: bytes


@dataclass
class GgufFile:
    path: str
    version: int
    kvs: list[KvRecord]
    tensors: list[TensorMeta]
    tensor_map: dict[str, TensorMeta]
    alignment: int
    data_offset: int


@dataclass
class PlanTensor:
    name: str
    dims: list[int]
    type: int
    size: int
    source: str | None = None
    generator: Callable[[], bytes] | None = None
    convert: str | None = None
    new_offset: int = 0


def align(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def product(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def tensor_nbytes(tensor_type: int, dims: list[int]) -> int:
    if tensor_type not in TYPE_BLOCKS:
        raise SystemExit(f"unsupported tensor type {tensor_type}")
    elems = product(dims)
    block_elems, block_bytes = TYPE_BLOCKS[tensor_type]
    blocks = (elems + block_elems - 1) // block_elems
    return blocks * block_bytes


def read_exact(fp: BinaryIO, n: int) -> bytes:
    data = fp.read(n)
    if len(data) != n:
        raise SystemExit("short GGUF read")
    return data


def read_u32(fp: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(fp, 4))[0]


def read_u64(fp: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(fp, 8))[0]


def write_u32(fp: BinaryIO, v: int) -> None:
    fp.write(struct.pack("<I", v))


def write_u64(fp: BinaryIO, v: int) -> None:
    fp.write(struct.pack("<Q", v))


def read_gguf_string(fp: BinaryIO) -> str:
    n = read_u64(fp)
    return read_exact(fp, n).decode("utf-8")


def write_gguf_string(fp: BinaryIO, s: str) -> None:
    b = s.encode("utf-8")
    write_u64(fp, len(b))
    fp.write(b)


def encode_gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_string_size(s: str) -> int:
    return 8 + len(s.encode("utf-8"))


def kv_string(key: str, value: str) -> KvRecord:
    raw = encode_gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + encode_gguf_string(value)
    return KvRecord(key=key, type=GGUF_TYPE_STRING, raw=raw)


def kv_u32(key: str, value: int) -> KvRecord:
    raw = encode_gguf_string(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)
    return KvRecord(key=key, type=GGUF_TYPE_UINT32, raw=raw)


def kv_f32(key: str, value: float) -> KvRecord:
    raw = encode_gguf_string(key) + struct.pack("<I", GGUF_TYPE_FLOAT32) + struct.pack("<f", value)
    return KvRecord(key=key, type=GGUF_TYPE_FLOAT32, raw=raw)


def kv_bool(key: str, value: bool) -> KvRecord:
    raw = encode_gguf_string(key) + struct.pack("<I", GGUF_TYPE_BOOL) + (b"\x01" if value else b"\x00")
    return KvRecord(key=key, type=GGUF_TYPE_BOOL, raw=raw)


def scalar_size(value_type: int) -> int:
    if value_type in (0, 1, 7):
        return 1
    if value_type in (2, 3):
        return 2
    if value_type in (4, 5, 6):
        return 4
    if value_type in (10, 11, 12):
        return 8
    raise SystemExit(f"unsupported GGUF metadata scalar type {value_type}")


def skip_value(fp: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        n = read_u64(fp)
        fp.seek(n, os.SEEK_CUR)
        return
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = read_u32(fp)
        n = read_u64(fp)
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(n):
                slen = read_u64(fp)
                fp.seek(slen, os.SEEK_CUR)
        else:
            fp.seek(n * scalar_size(elem_type), os.SEEK_CUR)
        return
    fp.seek(scalar_size(value_type), os.SEEK_CUR)


def read_u32_value(raw: bytes) -> int:
    return struct.unpack_from("<I", raw, len(raw) - 4)[0]


def load_gguf(path: str) -> GgufFile:
    with open(path, "rb") as fp:
        if read_exact(fp, 4) != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        version = read_u32(fp)
        n_tensors = read_u64(fp)
        n_kv = read_u64(fp)

        kvs: list[KvRecord] = []
        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            start = fp.tell()
            key = read_gguf_string(fp)
            value_type = read_u32(fp)
            skip_value(fp, value_type)
            end = fp.tell()
            fp.seek(start)
            raw = read_exact(fp, end - start)
            kvs.append(KvRecord(key=key, type=value_type, raw=raw))
            if key == "general.alignment" and value_type == GGUF_TYPE_UINT32:
                alignment = read_u32_value(raw) or GGUF_DEFAULT_ALIGNMENT

        tensors: list[TensorMeta] = []
        for _ in range(n_tensors):
            name = read_gguf_string(fp)
            n_dims = read_u32(fp)
            dims = [read_u64(fp) for _ in range(n_dims)]
            tensor_type = read_u32(fp)
            old_offset = read_u64(fp)
            tensors.append(
                TensorMeta(
                    name=name,
                    dims=dims,
                    type=tensor_type,
                    old_offset=old_offset,
                    size=tensor_nbytes(tensor_type, dims),
                )
            )
        data_offset = align(fp.tell(), alignment)

    return GgufFile(
        path=path,
        version=version,
        kvs=kvs,
        tensors=tensors,
        tensor_map={t.name: t for t in tensors},
        alignment=alignment,
        data_offset=data_offset,
    )


def copied(src: GgufFile, out_name: str, in_name: str) -> PlanTensor:
    if in_name not in src.tensor_map:
        raise SystemExit(f"source tensor missing: {in_name}")
    t = src.tensor_map[in_name]
    return PlanTensor(
        name=out_name,
        dims=list(t.dims),
        type=t.type,
        size=t.size,
        source=in_name,
    )


def copied_f16_to_f32(src: GgufFile, out_name: str, in_name: str) -> PlanTensor:
    if in_name not in src.tensor_map:
        raise SystemExit(f"source tensor missing: {in_name}")
    t = src.tensor_map[in_name]
    if t.type != TENSOR_F16:
        return copied(src, out_name, in_name)
    return PlanTensor(
        name=out_name,
        dims=list(t.dims),
        type=TENSOR_F32,
        size=tensor_nbytes(TENSOR_F32, t.dims),
        source=in_name,
        convert="f16_to_f32",
    )


def f32_values(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def f32_constant(count: int, value: float) -> bytes:
    return struct.pack("<f", value) * count


def q8_0_projector(mode: str, scale: float, dim: int = DS4_N_EMBD) -> bytes:
    row_size = (dim // 32) * 34
    out = bytearray(row_size * dim)
    d_one = struct.pack("<e", scale)
    d_zero = struct.pack("<e", 0.0)
    for row in range(dim):
        row_off = row * row_size
        for block in range(dim // 32):
            off = row_off + block * 34
            if mode == "identity" and block == row // 32:
                out[off : off + 2] = d_one
                out[off + 2 + (row % 32)] = 1
            else:
                out[off : off + 2] = d_zero
    return bytes(out)


def generated(name: str, dims: list[int], tensor_type: int, gen: Callable[[], bytes]) -> PlanTensor:
    return PlanTensor(
        name=name,
        dims=dims,
        type=tensor_type,
        size=tensor_nbytes(tensor_type, dims),
        generator=gen,
    )


def build_plan(
    src: GgufFile,
    layer: int,
    e_proj: str,
    h_proj: str,
    e_scale: float,
    h_scale: float,
    norm_source: str,
    plain_f32: bool,
) -> list[PlanTensor]:
    blk = f"blk.{layer}"
    copy_plain = copied_f16_to_f32 if plain_f32 else copied
    plan: list[PlanTensor] = [
        copied(src, "mtp.0.hc_head_base.weight", "output_hc_base.weight"),
        copy_plain(src, "mtp.0.hc_head_fn.weight", "output_hc_fn.weight"),
        copied(src, "mtp.0.hc_head_scale.weight", "output_hc_scale.weight"),
        generated(
            "mtp.0.e_proj.weight",
            [DS4_N_EMBD, DS4_N_EMBD],
            TENSOR_Q8_0,
            lambda: q8_0_projector(e_proj, e_scale),
        ),
        generated(
            "mtp.0.h_proj.weight",
            [DS4_N_EMBD, DS4_N_EMBD],
            TENSOR_Q8_0,
            lambda: q8_0_projector(h_proj, h_scale),
        ),
    ]

    if norm_source == "output":
        plan.extend(
            [
                copied(src, "mtp.0.enorm.weight", "output_norm.weight"),
                copied(src, "mtp.0.hnorm.weight", "output_norm.weight"),
                copied(src, "mtp.0.norm.weight", "output_norm.weight"),
            ]
        )
    elif norm_source == "layer":
        plan.extend(
            [
                copied(src, "mtp.0.enorm.weight", f"{blk}.attn_norm.weight"),
                copied(src, "mtp.0.hnorm.weight", f"{blk}.attn_norm.weight"),
                copied(src, "mtp.0.norm.weight", "output_norm.weight"),
            ]
        )
    else:
        plan.extend(
            [
                generated("mtp.0.enorm.weight", [DS4_N_EMBD], TENSOR_F32, lambda: f32_constant(DS4_N_EMBD, 1.0)),
                generated("mtp.0.hnorm.weight", [DS4_N_EMBD], TENSOR_F32, lambda: f32_constant(DS4_N_EMBD, 1.0)),
                copied(src, "mtp.0.norm.weight", "output_norm.weight"),
            ]
        )

    suffixes = [
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_attn_base.weight",
        "attn_norm.weight",
        "attn_q_a.weight",
        "attn_q_a_norm.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_kv_a_norm.weight",
        "attn_sinks.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
        "hc_ffn_base.weight",
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ]
    for suffix in suffixes:
        in_name = f"{blk}.{suffix}"
        if plain_f32 and suffix in ("hc_attn_fn.weight", "hc_ffn_fn.weight", "ffn_gate_inp.weight"):
            plan.append(copied_f16_to_f32(src, f"mtp.0.{suffix}", in_name))
        else:
            plan.append(copied(src, f"mtp.0.{suffix}", in_name))

    bias_name = f"{blk}.exp_probs_b.bias"
    if bias_name in src.tensor_map:
        plan.append(copied(src, "mtp.0.exp_probs_b.bias", bias_name))
    else:
        plan.append(generated("mtp.0.exp_probs_b.bias", [DS4_N_EXPERT], TENSOR_F32, lambda: f32_constant(DS4_N_EXPERT, 0.0)))

    return plan


def prepare_offsets(plan: list[PlanTensor], alignment: int) -> int:
    off = 0
    for t in plan:
        t.new_offset = off
        off += align(t.size, alignment)
    return off


def tensor_info_size(plan: list[PlanTensor]) -> int:
    total = 0
    for t in plan:
        total += gguf_string_size(t.name) + 4 + len(t.dims) * 8 + 4 + 8
    return total


def copy_tensor_data(src: GgufFile, fp_src: BinaryIO, source_name: str) -> bytes:
    t = src.tensor_map[source_name]
    fp_src.seek(src.data_offset + t.old_offset)
    return read_exact(fp_src, t.size)


def f16_bytes_to_f32(data: bytes) -> bytes:
    out = bytearray((len(data) // 2) * 4)
    for i in range(0, len(data), 2):
        v = struct.unpack_from("<e", data, i)[0]
        struct.pack_into("<f", out, (i // 2) * 4, v)
    return bytes(out)


def write_padding(fp: BinaryIO, n: int) -> None:
    if n <= 0:
        return
    zeros = b"\0" * min(n, 1024 * 1024)
    while n:
        chunk = min(n, len(zeros))
        fp.write(zeros[:chunk])
        n -= chunk


def write_gguf(src: GgufFile, plan: list[PlanTensor], out_path: str, overwrite: bool, extra_kvs: list[KvRecord]) -> None:
    if os.path.exists(out_path) and not overwrite:
        raise SystemExit(f"output exists: {out_path} (use --overwrite)")

    tensor_bytes = prepare_offsets(plan, src.alignment)
    kvs = src.kvs + extra_kvs
    kv_raw = b"".join(kv.raw for kv in kvs)
    meta_size = 4 + 4 + 8 + 8 + len(kv_raw) + tensor_info_size(plan)
    out_data_offset = align(meta_size, src.alignment)

    tmp_path = out_path + ".tmp"
    with open(src.path, "rb") as fp_src, open(tmp_path, "wb") as fp_out:
        fp_out.write(GGUF_MAGIC)
        write_u32(fp_out, src.version)
        write_u64(fp_out, len(plan))
        write_u64(fp_out, len(kvs))
        fp_out.write(kv_raw)
        for t in plan:
            write_gguf_string(fp_out, t.name)
            write_u32(fp_out, len(t.dims))
            for dim in t.dims:
                write_u64(fp_out, dim)
            write_u32(fp_out, t.type)
            write_u64(fp_out, t.new_offset)

        pos = fp_out.tell()
        if pos > out_data_offset:
            raise SystemExit("metadata is larger than planned")
        write_padding(fp_out, out_data_offset - pos)

        for i, t in enumerate(plan, 1):
            print(
                f"[{i:2d}/{len(plan):2d}] {t.name} {TYPE_NAMES.get(t.type, t.type)} "
                f"{t.size / 1048576.0:.2f} MiB",
                file=sys.stderr,
            )
            if t.source:
                data = copy_tensor_data(src, fp_src, t.source)
                if t.convert == "f16_to_f32":
                    data = f16_bytes_to_f32(data)
            elif t.generator:
                data = t.generator()
            else:
                raise SystemExit(f"no data source for {t.name}")
            if len(data) != t.size:
                raise SystemExit(f"{t.name}: generated {len(data)} bytes, expected {t.size}")
            fp_out.write(data)
            write_padding(fp_out, align(t.size, src.alignment) - t.size)

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path}")
    print(f"tensor_bytes_unpadded={sum(t.size for t in plan)}")
    print(f"approx_file_bytes={out_data_offset + tensor_bytes}")


def print_plan(src: GgufFile, plan: list[PlanTensor], extra_kvs: list[KvRecord]) -> None:
    prepare_offsets(plan, src.alignment)
    print(f"source={src.path}")
    print(f"alignment={src.alignment}")
    print(f"kv_count={len(src.kvs) + len(extra_kvs)}")
    print(f"tensor_count={len(plan)}")
    print(f"tensor_bytes_unpadded={sum(t.size for t in plan)}")
    print(f"approx_file_bytes={align(4 + 4 + 8 + 8 + sum(len(kv.raw) for kv in src.kvs + extra_kvs) + tensor_info_size(plan), src.alignment) + sum(align(t.size, src.alignment) for t in plan)}")
    for t in plan:
        src_desc = t.source if t.source else "generated"
        print(f"{t.name}\t{TYPE_NAMES.get(t.type, t.type)}\t{t.dims}\t{src_desc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="ds4flash.gguf", help="target GGUF to bootstrap from")
    ap.add_argument("--out", required=True, help="output MTP support GGUF")
    ap.add_argument("--layer", type=int, default=1, help="target block copied into mtp.0.*, default 1")
    ap.add_argument("--e-proj", choices=("identity", "zero"), default="zero")
    ap.add_argument("--h-proj", choices=("identity", "zero"), default="identity")
    ap.add_argument("--e-scale", type=float, default=1.0, help="Q8_0 scale for identity e_proj")
    ap.add_argument("--h-scale", type=float, default=1.0, help="Q8_0 scale for identity h_proj")
    ap.add_argument("--norm-source", choices=("ones", "output", "layer"), default="ones")
    ap.add_argument("--plain-f32", action="store_true", help="upcast F16 plain mixer/router tensors to F32")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def bootstrap_kvs(args: argparse.Namespace) -> list[KvRecord]:
    return [
        kv_bool("ds4.mtp.bootstrap", True),
        kv_string("ds4.mtp.bootstrap.source", args.source),
        kv_u32("ds4.mtp.bootstrap.layer", args.layer),
        kv_string("ds4.mtp.bootstrap.e_proj", args.e_proj),
        kv_string("ds4.mtp.bootstrap.h_proj", args.h_proj),
        kv_f32("ds4.mtp.bootstrap.e_scale", args.e_scale),
        kv_f32("ds4.mtp.bootstrap.h_scale", args.h_scale),
        kv_string("ds4.mtp.bootstrap.norm_source", args.norm_source),
        kv_bool("ds4.mtp.bootstrap.plain_f32", args.plain_f32),
    ]


def main() -> int:
    args = parse_args()
    src = load_gguf(args.source)
    plan = build_plan(src, args.layer, args.e_proj, args.h_proj, args.e_scale, args.h_scale, args.norm_source, args.plain_f32)
    extra_kvs = bootstrap_kvs(args)
    if args.dry_run:
        print_plan(src, plan, extra_kvs)
        return 0
    write_gguf(src, plan, args.out, args.overwrite, extra_kvs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
