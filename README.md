# CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF local inference engine with M5 Metal support

This is **a personal fork of [antirez/ds4](https://github.com/antirez/ds4)**, the
DeepSeek V4 Flash inference engine. It exists as a staging area to combine
upstream pull requests that I want to run together on my own M5 hardware
before they land upstream.

`main` here is `antirez/ds4@HEAD` with three things merged in:

1. **`feat(loader): support stock-recipe (Q8_0/F32) GGUFs end-to-end on Metal`**
   ([branch `support-q8_0-token-embd`](https://github.com/audreyt/ds4/tree/support-q8_0-token-embd),
   sent as a PR to antirez/ds4). Makes ds4 accept GGUFs that were produced by
   the upstream llama.cpp converter without per-tensor type overrides — files
   where most small projections are Q8_0 and the routed-expert router is F32,
   instead of antirez's hand-tuned recipe where they all stay F16. The
   motivating case is the
   `cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF` models, but
   the change is generic.

2. **[ivanfioravanti's PR #15: Add Metal 4 M5 prefill optimizations](https://github.com/antirez/ds4/pull/15)**.
   M5-class Metal 4 (MPP) tensor-API paths for Q8_0 dense matmul, attention
   output low-projection, and staged routed-MoE projections, plus a fused
   six-expert routed-MoE sum kernel. ~1.5x prefill speedup on MacBook Pro M5
   Max for q2 prompts.

3. **`fix(metal): correct M5 MPP + Q8_0 ape compressor for stock-recipe GGUFs`**
   ([branch `m5-support-q8_0-token-embd`](https://github.com/audreyt/ds4/tree/m5-support-q8_0-token-embd)).
   Two fixes that close a regression where (1) + (2) together produce garbage
   output (BOS-token spam) for stock-recipe Q8_0 ape on M5: a CPU-side dequant
   for the prefill compressor APE byte-strided path (with per-call MTLBuffer
   to avoid an encode-time race condition on the shared scratch), and a Q8_0
   branch in the decode-time `kernel_dsv4_compressor_store_one` Metal kernel.
   See the m5-support-q8_0-token-embd commit message for the full story.

If you only want one of these, use the corresponding branch directly:

```
audreyt/ds4
├── main                          — all three merged
├── m5-support-q8_0-token-embd    — loader PR + ivan's PR #15 + the M5/cyber fix
├── support-q8_0-token-embd       — just the stock-recipe loader PR (the one I sent upstream)
└── (PR #15 lives at https://github.com/ivanfioravanti/ds4/tree/codex/metal4-m5-scaffold)
```

## Why this fork exists

I run [`pi-ds4`](https://github.com/audreyt/pi-ds4) on a MacBook M5 Max, and
I wanted to try the cyberneurova abliterated DeepSeek V4 Flash GGUFs without
pre-converting the file or running a separate inference engine. Loading those
GGUFs into stock antirez/ds4 fails immediately because the recipe is different
— ds4 expects `token_embd.weight` to be F16, the cyberneurova file has it as
Q8_0; same for `output_hc_fn`, the HC fn weights, the attention/indexer
compressors, the indexer projections, and so on (~360 tensor mismatches across
12 families). The `support-q8_0-token-embd` PR is the engine work that closes
those gaps so the file just loads.

PR #15 is independent: it's M5-class throughput. I wanted both, but loading
unmodified cyberneurova on top of PR #15 surfaced a Q8_0-ape compressor bug
(also fixed in this fork's `m5-support-q8_0-token-embd` branch).

## What's verified on M5

| GGUF | flags | result |
|---|---|---|
| antirez recipe (q2 / q4) | defaults | works as upstream |
| cyberneurova `*-Q2_K.gguf` | defaults | works end-to-end including PR #15's MPP F16 prefill |

## Benchmarks

Prefill throughput on MacBook Pro M5 Max with the cyberneurova Q2_K GGUF
(`cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf`, ~95 GB), `--ctx 32768`,
3 repeats averaged. "MPP off" sets `DS4_METAL_MPP_DISABLE=1` (effectively the
non-M5 path); "MPP on" is the default for `audreyt/ds4` `main`, which
includes ivan's PR #15 plus the M5/cyber compressor fix. Same command shape
ivan used in PR #15: `./ds4 --prompt-file <prompt> -n 1 --nothink --ctx 32768`.

| prompt tokens | MPP off avg tok/s | MPP on avg tok/s | speedup |
|---:|---:|---:|---:|
| 533 | 261.6 | 442.4 | 1.69x |
| 2008 | 362.4 | 625.5 | 1.73x |
| 4107 | 308.4 | 553.8 | 1.80x |
| 8126 | 279.6 | 387.1 | 1.38x |
| 16300 | 273.2 | 413.8 | 1.51x |

Decode throughput sits around 24-37 tok/s and doesn't change in any
consistent direction with MPP, which matches PR #15's design (MPP is a
prefill-only optimization). Prompts were built by concatenating this
README's text and trimming to approximate target token counts.

## Build and run

Build is unchanged from upstream:

```sh
make
```

To run against an unmodified cyberneurova GGUF on M5:

```sh
ln -sfn /path/to/cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf ./ds4flash.gguf
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

To run against an antirez-recipe q2/q4 file, the upstream `download_model.sh`
flow works as-is:

```sh
./download_model.sh q2
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

## Acknowledgements

* **Salvatore Sanfilippo (antirez)** for [ds4](https://github.com/antirez/ds4)
  and the [llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash)
  converter that both this fork and the cyberneurova GGUFs depend on.
* **Ivan Fioravanti (ivanfioravanti)** for the M5 Metal 4 / MPP optimization
  work in PR #15.
* **Georgi Gerganov and the llama.cpp / GGML community** for the GGUF format,
  Metal kernel infrastructure, and quantization formats that all of this is
  built on. ds4's `LICENSE` retains the GGML copyright notice for that reason.
* **The cyberneurova research project** for publishing
  [the DeepSeek-V4-Flash abliterated GGUFs](https://huggingface.co/cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF)
  in the stock llama.cpp recipe — the motivating case for the loader PR.

The original upstream README (project design philosophy, model card,
server/CLI documentation, disk KV cache format, test vectors) lives at
[antirez/ds4#readme](https://github.com/antirez/ds4#readme). I haven't
duplicated it here so this file stays focused on what's *different* about
this fork.

## License

MIT, matching upstream. See `LICENSE`.
