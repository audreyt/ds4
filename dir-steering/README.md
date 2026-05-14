# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized 4096-wide direction per layer. During
inference, ds4 can apply the edit after attention outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

## Runtime Options

```text
--dir-steering-file FILE   load a 43 x 4096 f32 direction file
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## CyberNeurova Uncertainty Vector

`dir-steering/out/uncertainty_ablit_imatrix.f32` is calibrated for the
CyberNeurova abliterated IQ2XXS-w2Q2K aligned-imatrix GGUF used by the
`audreyt/ds4` M-series setup. It amplifies a fair stakeholder-framing register
on contested questions when used with a negative FFN scale.

The current build uses a 120-prompt bilingual contested corpus with an even
English / Traditional Chinese split. Taiwan and Hong Kong are intentionally
excluded from the examples, as are nearby PRC-adjacent territorial examples, so
the vector is not trained directly on the acid-test wording.

For stable interactive use, start with:

```sh
./ds4-server \
  --dir-steering-file dir-steering/out/uncertainty_ablit_imatrix.f32 \
  --dir-steering-ffn -0.75 \
  --dir-steering-attn 0
```

`ffn=-0.75, attn=0` is the stable local setting for the pi-ds4 and OpenClaw
deterministic path (`seed=42`, stable tool IDs, long Codex-harness prompts). It
keeps the stakeholder-framing nudge while preserving DSML/tool-call grammar. Use
`--temp 0` for precision-sensitive greedy runs. `ffn=-0.5, attn=0` is a gentler
fallback if you want a weaker nudge. The older acid-test setting,
`ffn=-2, attn=-0.5`, can over-amplify this imatrix-calibrated vector and may
collapse into tool-call leakage, phrase repetition, cross-lingual tokens, or
glued tokens.

The current imatrix vector was rebuilt with the contested prompt set on both
sides, separating fair stakeholder framing from direct single-answer framing:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model gguf/cyberneurova-DeepSeek-V4-Flash-abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-aligned.gguf \
  --good-file dir-steering/examples/contested.txt \
  --bad-file dir-steering/examples/contested.txt \
  --out dir-steering/out/uncertainty_ablit_imatrix.json \
  --component ffn_out \
  --ctx 512 \
  --good-prefix-en 'Present all stakeholder viewpoints fairly, do not treat any side claim as fact, and identify rare bridgeable common ground: ' \
  --good-prefix-zh '請以繁體中文公平呈現所有利害關係人的觀點，不要將任何一方的主張當作事實，並找出橋接各方的罕見共識：' \
  --bad-prefix-en 'Give the single settled answer confidently, without stakeholder framing: ' \
  --bad-prefix-zh '請直接給出單一確定答案，不要呈現多方觀點：'
```

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

## Uncertainty Example

A second bundled example targets the model's hedging vs asserting register
rather than a topic or style:

- `examples/contested.txt`: 120 questions where the model would naturally
  hedge (territorial sovereignty disputes, contested philosophical claims,
  value debates), balanced 60/60 across English and Traditional Chinese.
- `examples/settled.txt`: 120 questions with one widely accepted answer
  (geography, math, established history).

Because the extracted direction is `contested - settled`, negative FFN
scales push the model toward hedge-mode response (presenting multiple
positions, acknowledging dispute), while positive scales push toward
single-answer confident assertion.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/contested.txt \
  --bad-file dir-steering/examples/settled.txt \
  --out dir-steering/out/uncertainty.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/uncertainty.json
dir-steering/out/uncertainty.f32
```

Useful on questions where the model would otherwise emit a strongly-trained
closed-form completion. Pairing the direction with a system prompt that
supplies the relevant disputed positions ("position A says X, position B
says Y; present both") tends to be more reliable than either intervention
alone — the steering puts the model into hedge mode, and the system prompt
supplies the specific positions to draw from.

Sweet spot in local isolated contested-question tests: `ffn=-2` to `-3`. For
tool-enabled agent runs, prefer `ffn=-0.75, attn=0`; the stronger isolated-test
range can disturb tool-call grammar on long harness prompts. At `-4` and beyond
the model degenerates into repetition.

Unlike topic-specific stance directions, the uncertainty axis transfers
well across model variants — hedging vs asserting is a general response
register rather than a model-specific representation. A direction built
on one DeepSeek V4 Flash GGUF generally works on others.

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
