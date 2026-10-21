# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized hidden-width direction per normal
transformer layer. During inference, ds4 can apply the edit after attention
outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

The file shape depends on the model:

- DeepSeek V4 Flash: `43 x 4096`.
- DeepSeek V4.1 Flash: `40 x 5120`. The FFN scale edits the post-layer
  residual across the hyper-connection copies (the GLP refusal-vector site);
  the attention scale edits the attention block output. Dumps average those
  copies at the last prompt token, at the same site.
- GLM 5.3 Flash: `45 x 4096`. The separate MTP predictor layer is omitted.
- Qwen3.8 Flash Next: `48 x 2560`. FFN steering is applied to each
  hyper-connection branch of the residual; dumps average those branches
  at the last prompt token.

GLM 5.2 steering is not implemented.

## Runtime Options

```text
--dir-steering-file FILE   load one f32 direction per normal model layer
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## GLM 5.3 Example

Build a GLM 5.3 direction from paired target and control prompt lists:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile glm-5.3-flash \
  --ds4 ./ds4 \
  --model gguf/GLM-5.3-Flash-Q2.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/glm53-direction.json \
  --component ffn_out \
  --ctx 512
```

Generated `.f32` vectors are local artifacts and are not stored in the
repository. GLM 5.3 steering works with `--mtp`, `ds4-server`, native session
batching, and two-Mac tensor parallelism. For tensor parallelism, pass the same
steering file and scales to both the worker and coordinator.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile deepseek-v4-flash \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.

## Qwen3.8 Flash Next

Capture uses `--think` / `--nothink`
(not `--think-high`). Dumps track the prompt phase explicitly, including
one-token tails, and retain the last prompt token during ordinary and MTP
decode. `attn_out` captures the output projection of both GDN and full-attention
layers, giving one row for each of the 48 trunk layers:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile qwen3.8-flash-next \
  --ds4 ./ds4 \
  --model gguf/Qwen3.8-Flash-Next-Q4.gguf \
  --good-file /path/to/target-prompts.txt \
  --bad-file /path/to/control-prompts.txt \
  --out dir-steering/out/qwen38-direction.json \
  --component ffn_out \
  --ctx 512
```

Qwen steering is Metal-only. `--mtp-model`, SSD streaming, and `--power`
remain unsupported for this graph. The bank contains only the 48 trunk layers;
the embedded MTP predictor remains unsteered. Its drafts are verified by the
steered target trunk, so `--mtp` remains supported.

## DeepSeek V4.1 Flash: a candor direction

`out/candor-v41.f32` is a V4.1 direction for questions on politically
sensitive history, where a model may answer with a refusal, an evasion or a
single account presented as settled instead of the evidence. It follows the
contrast-pair recipe zjj1993 published for Qwen3.8 Flash Next on Hugging
Face: 30 pairs, in English and Chinese, each putting a sensitive event next
to a comparable event elsewhere, asked the same way. The direction is the
per-layer mean of the sensitive prompts minus the mean of their comparisons,
normalized and not orthogonalized. At scale 1 the FFN edit projects it out
of every hyper-connection copy, `h - (h.d)d`, for prompt and generated tokens
alike.

The prompt files are under the Qwen Community License 1.0 and are not copied
here. They wrap each question in Qwen's chat template, while ds4 applies its
own, so keep only the user turn:

```sh
for f in pos neg held; do
  curl -fsSL https://huggingface.co/zjj1993/Qwen3.8-Flash-Next-uncensored-china/resolve/main/data/$f.txt |
    python3 -c 'import re, sys
for line in sys.stdin:
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", line.replace("\\n", "\n"), re.S)
    if m: print(m.group(1).strip())' > /tmp/candor-$f.txt
done
```

Build from the unabliterated weights, capturing at the start of thinking, and
keep layers 3 to 36. The source kept 4 to 44 of 48; this window likewise
leaves the last layers, next to the logits, untouched:

```sh
python3 dir-steering/tools/build_direction.py \
  --profile deepseek-v41-flash \
  --ds4 ./ds4 \
  --model gguf/DeepSeek-V4.1-Flash-1017-Q1.gguf \
  --good-file /tmp/candor-pos.txt \
  --bad-file /tmp/candor-neg.txt \
  --out dir-steering/out/candor-v41.json \
  --component ffn_out \
  --think --no-orthogonalize --keep-layers 3:36
```

Run it at scale 1. Larger scales flip the component instead of removing it,
and on V4.1 scale 2 added factual errors without reducing evasion further:

```sh
./ds4-server --dir-steering-file dir-steering/out/candor-v41.f32 --dir-steering-ffn 1
```

On 68 held-out prompts (48 on sensitive history, 20 comparisons about other
places, half in each language), each layer's direction separates the two
groups at AUC 0.84 to 0.93 from layer 12 on. Greedy, non-thinking answers
from the abliterated Q1, graded into the source's categories; the last
column adds the system prompt `Present fairly all stakeholder perspectives
and the uncommon ground that bridges them.`:

| Sensitive-history prompts (48) | scale 0 | scale 1 | scale 1 + prompt |
|---|---|---|---|
| evasive (refusal, feigned ignorance, one account as settled) | 23 | 9 | 3 |
| calibrated (substantive, says where evidence is disputed) | 14 | 25 | 33 |
| one-sided | 8 | 13 | 8 |
| major factual error | 3 | 1 | 4 |

The comparison prompts stayed at 16 of 20 calibrated at scale 1 (17, with
3 factual errors, with the prompt added), and 20 short arithmetic, fact and
code checks stayed correct. The projection removes the evasion without
pushing the model the other way, so some former refusals come back
one-sided, and memorised one-line answers to sovereignty questions survive
the direction alone. With the system prompt as well, both sovereignty test
questions came back with the positions laid out side by side. With thinking
on the results point the same way but are sampled and noisier, and about a
quarter of the sensitive prompts thought until a 3000-token cap, steered or
not.
