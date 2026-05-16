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
--dir-steering-policy MODE server-only policy: final-answer, decoding, always, or off; default is final-answer
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

For tool-using agents, `ds4-server` defaults to `--dir-steering-policy
final-answer`. This keeps prompt prefill, thinking tokens, and DSML tool-call
tokens unsteered. Steering is re-enabled only after generation has clearly
entered final natural-language answer text. This avoids letting a
behavior/style vector perturb tool-call grammar while still allowing the final
prose to use the configured direction.

`--dir-steering-policy decoding` is a middle ground for experiments that should
leave prompt/prefill activations untouched but steer every generated token,
including thinking and tool-call syntax. `always` restores the original
always-on behavior, and `off` disables directional steering at the server policy
layer.

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

## Doc-to-Steering Prototype

`tools/build_doc_direction.py` generates paired target/contrast probes from a
document set, then can call `build_direction.py` to capture a normal DS4
direction. The target probes include short reference excerpts and ask for a
DS4-maintainer answer. The contrast probes ask for the same kind of answer in
generic terms without repository-local details.

Generate prompt sidecars only:

```sh
python3 dir-steering/tools/build_doc_direction.py \
  --doc README.md \
  --doc MODEL_CARD.md \
  --doc AGENT.md \
  --out dir-steering/out/ds4_docs.json \
  --max-prompts 64
```

This writes:

```text
dir-steering/out/ds4_docs.doc-good.txt
dir-steering/out/ds4_docs.doc-bad.txt
dir-steering/out/ds4_docs.doc-to-steering.json
```

Capture the steering vector by adding `--build`:

```sh
python3 dir-steering/tools/build_doc_direction.py \
  --doc README.md \
  --doc MODEL_CARD.md \
  --doc AGENT.md \
  --out dir-steering/out/ds4_docs.json \
  --max-prompts 64 \
  --ctx 768 \
  --component ffn_out \
  --build
```

Then try it with a gentle negative FFN scale:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/ds4_docs.f32 \
  --dir-steering-ffn -0.5 \
  -p "How should I debug a DS4 tool-call regression?"
```

This is a document-conditioned register and behavior nudge, not a fact store.
Use it to make answers lean toward the corpus' terminology, maintenance
constraints, and debugging posture. Pair it with normal context, retrieval, or
KV-cache reuse when exact document facts matter.

## Doc-to-Steering Hypernetwork

`tools/doc_steering_hypernetwork.py` trains a tiny dependency-free
learned-basis hypernetwork from captured steering examples. It maps document
text to learned coefficients over known `.f32` directions, mixes those
directions, and normalizes the result layer by layer. The output layer is still
a basis of DS4 steering vectors, but the text-to-coefficients mapping is trained
with a sparse softmax model rather than selected by nearest neighbor.

Train it on one or more recipes produced by the doc-to-steering builder:

```sh
python3 dir-steering/tools/doc_steering_hypernetwork.py train \
  --recipe dir-steering/out \
  --model-dir dir-steering/out/doc-hnet \
  --kind learned-basis \
  --top-k 4 \
  --epochs 240
```

Predict a new steering file:

```sh
python3 dir-steering/tools/doc_steering_hypernetwork.py predict \
  --model-dir dir-steering/out/doc-hnet \
  --doc README.md \
  --doc MODEL_CARD.md \
  --out dir-steering/out/predicted_docs.f32
```

Evaluate reconstruction, or leave-one-out generalization when there are enough
examples:

```sh
python3 dir-steering/tools/doc_steering_hypernetwork.py eval \
  --model-dir dir-steering/out/doc-hnet

python3 dir-steering/tools/doc_steering_hypernetwork.py eval \
  --model-dir dir-steering/out/doc-hnet \
  --leave-one-out
```

This first hypernetwork is a real trained coefficient model, but deliberately
small: no Torch dependency, no backprop through DS4, and no attempt to invent
directions outside the captured basis. Its job is to make the interface
reliable: recipes in, model directory out, document text in, normalized DS4
steering vector out. Once a larger corpus of successful document directions
exists, the same train/eval split can be used to replace the encoder with a
larger neural mapper.

Do not evaluate this by asking a no-context factual question and expecting the
steering vector to recite file names or commands from the source document.
Steering can bias register, terminology, and answer posture, but it is not a
reliable document memory. For factual doc questions, put the relevant excerpts
in context and use the predicted steering vector as a companion nudge.

## Context-Augmented Lore

`tools/lore_cag.py` is the factual-memory layer for document-conditioned
steering. It builds a cited lore pack, retrieves exact excerpts for a query,
composes a guarded prompt, and can run `./ds4` with either a fixed steering file
or a predicted hypernetwork steering file.

Build a lore pack:

```sh
python3 dir-steering/tools/lore_cag.py pack \
  --doc README.md \
  --doc MODEL_CARD.md \
  --doc AGENT.md \
  --doc dir-steering \
  --out dir-steering/out/ds4_lore.jsonl
```

For dated long-lore directories, filter by source date and keep adjacent chunks
around each retrieval hit:

```sh
python3 dir-steering/tools/lore_cag.py pack \
  --doc /Users/au/w/transcript \
  --out dir-steering/out/audrey-transcript-lore.jsonl \
  --after 2024-05-20 \
  --before 2026-05-16 \
  --max-chunk-chars 2200 \
  --min-chunk-chars 160
```

For large packs, build the persistent SQLite sidecar once. `retrieve`, `prompt`,
and `run` will automatically use `PACK.sqlite` when it exists, and will
auto-build it for packs larger than 32 MiB.

```sh
python3 dir-steering/tools/lore_cag.py index \
  --pack dir-steering/out/audrey-transcript-lore.jsonl
```

Inspect retrieval:

```sh
python3 dir-steering/tools/lore_cag.py retrieve \
  --pack dir-steering/out/ds4_lore.jsonl \
  --query "How do I build and test a directional-steering vector?" \
  --top-k 5 \
  --neighbor-chunks 1
```

Write the cited prompt without running the model:

```sh
python3 dir-steering/tools/lore_cag.py prompt \
  --pack dir-steering/out/ds4_lore.jsonl \
  --query "How do I build and test a directional-steering vector?" \
  --out /tmp/ds4_lore_prompt.txt
```

Run CAG through DS4:

```sh
python3 dir-steering/tools/lore_cag.py run \
  --pack dir-steering/out/ds4_lore.jsonl \
  --query "How do I build and test a directional-steering vector?" \
  --tokens 220 \
  --out-prompt /tmp/ds4_lore_prompt.txt
```

Run CAG plus the document-steering hypernetwork:

```sh
python3 dir-steering/tools/lore_cag.py run \
  --pack dir-steering/out/ds4_lore.jsonl \
  --query "How do I build and test a directional-steering vector?" \
  --hnet-model-dir dir-steering/out/doc-hnet \
  --dir-steering-ffn -0.5 \
  --tokens 220
```

The prompt tells the model to cite bracketed lore excerpts for concrete repo
facts and to say when the supplied lore does not contain an answer. This is the
part that prevents invented paths such as nonexistent steering scripts. The
steering vector still helps with register; the lore excerpts carry the facts.

Long-lore retrieval is date-aware and CJK-aware. Pack records include source
dates and titles, hidden worktree directories are excluded by default, and
retrieval can use MMR diversity plus neighbor chunks so a single long transcript
does not monopolize the prompt or lose the paragraph next to the exact hit.

## Lore Evaluation Loop

`tools/lore_cag_eval.py` sweeps lore retrieval and steering settings, runs DS4,
and scores each answer for expected source-backed terms, forbidden hallucinated
terms, and citations.

Create a deterministic test set from a [dated corpus](https://github.com/audreyt/transcript):

```sh
python3 dir-steering/tools/build_lore_testset.py \
  --doc /Users/au/w/transcript \
  --out dir-steering/out/audrey-transcript-cases.json \
  --after 2024-05-20 \
  --before 2026-05-16 \
  --limit 32
```

The generated cases quote short "needle" passages from selected transcripts and
score the model on returning the source date/title with citations. The expected
source fields are intentionally not included in the query, so compose-only mode
checks retrieval rather than merely rediscovering words from the question.

Compose-only retrieval coverage check:

```sh
python3 dir-steering/tools/lore_cag_eval.py \
  --pack dir-steering/out/ds4_lore.jsonl \
  --out dir-steering/out/cag-eval-compose.jsonl \
  --top-k 4,6 \
  --max-context-chars 7000,10000 \
  --neighbor-chunks 1 \
  --scales none \
  --compose-only
```

Long-lore live source-identification check:

```sh
python3 dir-steering/tools/lore_cag_eval.py \
  --pack dir-steering/out/audrey-transcript-lore.jsonl \
  --case-file dir-steering/out/audrey-transcript-cases.json \
  --out dir-steering/out/audrey-cag-eval-live.jsonl \
  --top-k 3 \
  --max-context-chars 8000 \
  --neighbor-chunks 1 \
  --scales none \
  --tokens 180
```

Train a lightweight outcome policy from the sweep rows:

```sh
python3 dir-steering/tools/train_lore_policy.py train \
  --outcome dir-steering/out/audrey-cag-eval-live.jsonl \
  --model-dir dir-steering/out/audrey-policy-live \
  --leave-one-case-out
```

Predict a runtime policy for a new query:

```sh
python3 dir-steering/tools/train_lore_policy.py predict \
  --model-dir dir-steering/out/audrey-policy-live \
  --query "Which transcript discusses local Kami systems with charters?"
```

The policy model is intentionally small and outcome-trained: it scores candidate
retrieval/steering configurations from the JSONL rows produced by
`lore_cag_eval.py`. Use compose-only rows to learn retrieval shape, and live
rows to learn whether CAG-only or CAG plus steering actually improves answers.
Do not treat this policy as a factual memory or as a replacement for live evals;
it is the routing layer that decides which CAG/steering setting deserves to run.

Live steering-strength sweep:

```sh
python3 dir-steering/tools/lore_cag_eval.py \
  --pack dir-steering/out/ds4_lore.jsonl \
  --out dir-steering/out/cag-eval-live.jsonl \
  --top-k 4 \
  --max-context-chars 7000 \
  --scales none,-0.25,-0.5,-1.0 \
  --hnet-model-dir dir-steering/out/doc-hnet \
  --tokens 260
```

Filter to selected cases:

```sh
python3 dir-steering/tools/lore_cag_eval.py \
  --pack dir-steering/out/ds4_lore.jsonl \
  --out dir-steering/out/cag-eval-metal.jsonl \
  --case metal_drift \
  --scales none,-1.0 \
  --hnet-model-dir dir-steering/out/doc-hnet
```

Current local seed results with five captured basis directions show that the
best setting is case-dependent: `steering_build_test` improved at `ffn=-1.0`,
`server_api` improved slightly at `ffn=-1.0`, while `quality_vectors` and
`metal_drift` preferred no steering. Treat the scale as a policy selected by
evaluation, not a global constant.
