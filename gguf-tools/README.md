# DS4 GGUF Tools

This directory contains the offline tools used to build and evaluate DeepSeek
V4 Flash GGUF files for `ds4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the DS4 output formats we actually ship:
  `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `make_mtp_draft_gguf.py`: bootstrap an MTP-only support GGUF from a target
  GGUF by renaming one target block into the `mtp.0.*` layout and synthesizing
  the MTP projector tensors.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ds4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Build A Bootstrap MTP GGUF

`make_mtp_draft_gguf.py` creates a valid ds4 MTP support GGUF directly from the
target GGUF. This is not a trained MTP module; it is a scratch bootstrap that
copies one target transformer block into the `mtp.0.*` names, copies the target
HC output head, and generates Q8_0 projector matrices.

The best current scratch preset is layer 1 with the token embedding projector
disabled and the previous-HC projector set to identity:

```sh
gguf-tools/make_mtp_draft_gguf.py \
  --source ds4flash.gguf \
  --out gguf/cyberneurova-mtp-bootstrap-l1-hproj.gguf \
  --layer 1 \
  --e-proj zero \
  --h-proj identity \
  --overwrite
```

This preset is useful for first-draft and verifier experiments because the
target HC already contains the current token. It is not a stable unchecked
multi-token turbo drafter: recursive bursts need a trained or distilled MTP
module with a real token-conditioned transition.

Generated files include `ds4.mtp.bootstrap.*` metadata keys recording the source
GGUF, copied layer, projector recipe, scales, norm source, and whether plain
F16 tensors were upcast to F32. Use `--plain-f32` to match the trained MTP's
precision profile for HC mixer and router-input tensors; it is useful for
experiments, but it does not replace actual distillation. ds4 detects the
`ds4.mtp.bootstrap` marker and disables unchecked turbo for these files by
default; set `DS4_MTP_BOOTSTRAP_TURBO=1` only when intentionally studying the
unstable recursive path.

## Dump MTP Distillation Records

`ds4 --mtp-distill-out` writes a compact binary dataset for training or auditing
MTP draft modules. Each record is a target-model step: the token just accepted,
the target HC state after that token, and the next-token top-k logits/logprobs.
This is the useful supervised signal for fitting a drafter to the exact target
model instead of tuning recursive turbo by eye.

```sh
./ds4 \
  -m ds4flash.gguf \
  -p "Write a concise explanation of speculative decoding." \
  --nothink \
  --mtp-distill-out /tmp/ds4-mtp-distill.bin \
  --mtp-distill-records 256 \
  --mtp-distill-top-k 32

gguf-tools/inspect_mtp_distill.py /tmp/ds4-mtp-distill.bin --records 3
```

The dump format starts with a 64-byte little-endian header (`DS4MTPD1`), then
fixed-size records. The header stores `n_hc`, `n_embd`, `vocab`, `top_k`,
`record_bytes`, and the finalized record count. Record payloads are
`pos/token/target/top_k/target_logit/target_logprob`, followed by `4 * 4096`
float32 HC values and `top_k` `(token, logit, logprob)` triples.

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the DS4 runtime:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this DS4 quantizer.  Generic GGUF
tools need DS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `ds4 --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the DS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Q4 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

Useful checks before writing a full model:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real DS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 gguf-tools/quality-testing/collect_official.py
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf gguf-tools/quality-testing/data/manifest.tsv /tmp/model.tsv 4096
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```
