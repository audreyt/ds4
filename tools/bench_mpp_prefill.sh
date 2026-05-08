#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4_BIN="$ROOT_DIR/ds4"
PROMPT_SOURCE="$ROOT_DIR/README.md"
SIZES="512,2048,4096,8192"
REPEATS=3
CTX=32768
GEN_TOKENS=1
MODEL=""
OUT_DIR=""
BUILD=1
INCLUDE_F16=0
CHARS_PER_TOKEN=4

usage() {
    cat <<'EOF'
Usage: tools/bench_mpp_prefill.sh [options]

Runs repeatable ds4 prefill benchmarks for legacy Metal and Metal 4 / MPP Q8_0
prefill.

Options:
  --prompt-source FILE   Source text used to synthesize prompts. Default: README.md
  --sizes LIST          Comma-separated approximate prompt-token sizes. Default: 512,2048,4096,8192
  --repeats N           Runs per variant and size. Default: 3
  --ctx N               ds4 context size. Default: 32768
  -n, --tokens N        Generated tokens per run. Default: 1
  -m, --model FILE      Model path passed to ds4.
  --out DIR             Output directory for prompts, logs, and results.csv. Default: /tmp
  --no-build            Do not run make ds4 before benchmarking.
  --include-f16         Also run DS4_METAL_MPP_EXPERIMENTAL_F16=1. Known graph-test unsafe.
  -h, --help            Show this help.
EOF
}

die() {
    echo "bench_mpp_prefill: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-source)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            PROMPT_SOURCE="$2"
            shift 2
            ;;
        --sizes)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            SIZES="$2"
            shift 2
            ;;
        --repeats)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            REPEATS="$2"
            shift 2
            ;;
        --ctx)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            CTX="$2"
            shift 2
            ;;
        -n|--tokens)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            GEN_TOKENS="$2"
            shift 2
            ;;
        -m|--model)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            MODEL="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            OUT_DIR="$2"
            shift 2
            ;;
        --no-build)
            BUILD=0
            shift
            ;;
        --include-f16)
            INCLUDE_F16=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -f "$PROMPT_SOURCE" ]] || die "prompt source not found: $PROMPT_SOURCE"
[[ "$REPEATS" =~ ^[0-9]+$ && "$REPEATS" -gt 0 ]] || die "--repeats must be a positive integer"
[[ "$CTX" =~ ^[0-9]+$ && "$CTX" -gt 0 ]] || die "--ctx must be a positive integer"
[[ "$GEN_TOKENS" =~ ^[0-9]+$ && "$GEN_TOKENS" -gt 0 ]] || die "--tokens must be a positive integer"

if [[ "$BUILD" -eq 1 ]]; then
    make -C "$ROOT_DIR" ds4 >/dev/null
fi
[[ -x "$DS4_BIN" ]] || die "ds4 binary not found or not executable: $DS4_BIN"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ds4-mpp-prefill.XXXXXX")"
else
    mkdir -p "$OUT_DIR"
fi

CSV="$OUT_DIR/results.csv"
printf 'variant,target_tokens,input_tokens,run,prefill_tps,generation_tps,status,log\n' > "$CSV"

IFS=',' read -r -a SIZE_LIST <<< "$SIZES"

make_prompt() {
    local target_tokens="$1"
    local out="$2"
    [[ "$target_tokens" =~ ^[0-9]+$ && "$target_tokens" -gt 0 ]] ||
        die "invalid size in --sizes: $target_tokens"

    local target_chars=$((target_tokens * CHARS_PER_TOKEN))
    : > "$out.full"
    while [[ "$(wc -c < "$out.full" | tr -d ' ')" -lt "$target_chars" ]]; do
        cat "$PROMPT_SOURCE" >> "$out.full"
        printf '\n' >> "$out.full"
    done
    head -c "$target_chars" "$out.full" > "$out"
    rm -f "$out.full"
}

run_one() {
    local variant="$1"
    local target_tokens="$2"
    local run="$3"
    local prompt="$4"
    local log="$OUT_DIR/${variant}_target${target_tokens}_run${run}.log"
    local status="ok"

    local cmd=("$DS4_BIN" --prompt-file "$prompt" -n "$GEN_TOKENS" --nothink --ctx "$CTX")
    if [[ -n "$MODEL" ]]; then
        cmd+=(-m "$MODEL")
    fi

    case "$variant" in
        legacy)
            if ! env DS4_METAL_MPP_DISABLE=1 "${cmd[@]}" >"$log" 2>&1; then status="fail"; fi
            ;;
        q8_mpp)
            if ! env DS4_METAL_MPP_ENABLE=1 "${cmd[@]}" >"$log" 2>&1; then status="fail"; fi
            ;;
        f16_mpp_unsafe)
            if ! env DS4_METAL_MPP_EXPERIMENTAL_F16=1 "${cmd[@]}" >"$log" 2>&1; then status="fail"; fi
            ;;
        *)
            die "internal unknown variant: $variant"
            ;;
    esac

    local input_tokens prefill generation
    input_tokens="$(sed -n 's/^processing \([0-9][0-9]*\) input tokens:.*/\1/p' "$log" | tail -n 1)"
    prefill="$(awk '/ds4: prefill:/ { for (i = 1; i <= NF; i++) if ($i == "prefill:") { print $(i + 1); exit } }' "$log")"
    generation="$(awk '/ds4: prefill:/ { for (i = 1; i <= NF; i++) if ($i == "generation:") { print $(i + 1); exit } }' "$log")"
    [[ -n "$input_tokens" ]] || input_tokens="$target_tokens"
    [[ -n "$prefill" ]] || prefill="NA"
    [[ -n "$generation" ]] || generation="NA"

    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$variant" "$target_tokens" "$input_tokens" "$run" "$prefill" "$generation" "$status" "$log" |
        tee -a "$CSV"
}

echo "bench_mpp_prefill: output directory: $OUT_DIR" >&2
echo "bench_mpp_prefill: csv: $CSV" >&2
echo "variant,target_tokens,input_tokens,run,prefill_tps,generation_tps,status,log"

for target in "${SIZE_LIST[@]}"; do
    prompt="$OUT_DIR/prompt_target${target}.txt"
    make_prompt "$target" "$prompt"
    for run in $(seq 1 "$REPEATS"); do
        run_one legacy "$target" "$run" "$prompt"
        run_one q8_mpp "$target" "$run" "$prompt"
        if [[ "$INCLUDE_F16" -eq 1 ]]; then
            run_one f16_mpp_unsafe "$target" "$run" "$prompt"
        fi
    done
done

echo
echo "Summary:"
printf "%-16s %-14s %-14s %8s %10s %10s %10s\n" "variant" "target_tokens" "input_tokens" "runs" "avg_tps" "min_tps" "max_tps"
awk -F, '
NR > 1 && $7 == "ok" && $5 != "NA" {
    key = $1 "," $2 "," $3
    sum[key] += $5
    n[key] += 1
    if (!(key in min) || $5 < min[key]) min[key] = $5
    if (!(key in max) || $5 > max[key]) max[key] = $5
}
END {
    for (key in n) {
        split(key, parts, ",")
        printf "%-16s %-14s %-14s %8d %10.2f %10.2f %10.2f\n", parts[1], parts[2], parts[3], n[key], sum[key]/n[key], min[key], max[key]
    }
}
' "$CSV" | sort
