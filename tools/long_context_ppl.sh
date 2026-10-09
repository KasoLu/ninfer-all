#!/usr/bin/env bash
# long_context_ppl.sh — long-context perplexity depth matrix on one long UTF-8 file.
#
# Matrix (same text file, same protocol, only --kv-dtype and the depth change):
#   bf16      @  64K   (deepest depth whose KV fits beside the 27B weights on a 24 GB card)
#   rk8v4     @ 128K, 200K
#   rk6v4-e8  @ 128K, 200K   (format under test; the 200K run is its capacity goal)
#
# bf16 and the quantized formats never share a depth: the bf16 row is the shallow reference,
# rk8v4 is the same-depth companion for the quantized formats. Any report built from these
# runs must state the depth asymmetry (same convention as the needle retrieval gate).
#
# Prerequisite (run first, must be green):
#   ctest --test-dir build -R 'i6_roundtrip|kv_cache_append' --output-on-failure
#
# usage: tools/long_context_ppl.sh <model.ninfer> <utf8-text-file> [options]
#   --dry-run                 print the commands without executing any run
#   --out DIR                 report root (default profiles/perplexity/long_ctx)
#   --context64 N --stride64 N    bf16 row (default 65536 / 32768)
#   --context128 N --stride128 N  (default 131072 / 65536)
#   --context200 N --stride200 N  (default 200000 / 100000)
#   --skip-bf16               omit the bf16 64K row
#
# Per-run reports land in <out>/<kv-dtype>.<context>/report.json. Existing per-run
# directories are removed before a re-run so the script is idempotent.

set -euo pipefail

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

root="$(cd "$(dirname "$0")/.." && pwd)"
perplexity=""
for candidate in "$root/build-ninja/apps/ninfer-perplexity" "$root/build/apps/ninfer-perplexity" "${NINFER_PERPLEXITY:-}"; do
  if [[ -f "$candidate" ]]; then perplexity="$candidate"; break
  elif [[ -f "$candidate.exe" ]]; then perplexity="$candidate.exe"; break
  fi
done
out="$root/profiles/perplexity/long_ctx"
dry_run=0
skip_bf16=0
ctx64=65536 stride64=32768
ctx128=131072 stride128=65536
ctx200=200000 stride200=100000
model="" text=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     dry_run=1 ;;
    --skip-bf16)   skip_bf16=1 ;;
    --out)         out="$2"; shift ;;
    --context64)   ctx64="$2"; shift ;;
    --stride64)    stride64="$2"; shift ;;
    --context128)  ctx128="$2"; shift ;;
    --stride128)   stride128="$2"; shift ;;
    --context200)  ctx200="$2"; shift ;;
    --stride200)   stride200="$2"; shift ;;
    --help|-h)     usage 0 ;;
    -*)            echo "unknown option: $1" >&2; usage 1 ;;
    *)             if [[ -z "$model" ]]; then model="$1"
                   elif [[ -z "$text" ]]; then text="$1"
                   else echo "unexpected argument: $1" >&2; usage 1
                   fi ;;
  esac
  shift
done

[[ -n "$model" && -n "$text" ]] || usage 1

py="$(command -v python3 || command -v python || true)"

# run <kv-dtype> <context> <stride>
run() {
  local kv="$1" context="$2" stride="$3"
  local dir="$out/$kv.$context"
  local cmd=("$perplexity" "$model" --text "$text"
             --context "$context" --stride "$stride"
             --kv-dtype "$kv" --output "$dir")

  if [[ "$dry_run" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$(printf '%q ' "${cmd[@]}")"
    return
  fi

  [[ -f "$perplexity" ]] || { echo "error: perplexity binary not found: $perplexity (run the build first)" >&2; exit 1; }
  [[ -f "$model" ]] || { echo "error: artifact not found: $model" >&2; exit 1; }
  [[ -f "$text" ]] || { echo "error: text file not found: $text" >&2; exit 1; }

  rm -rf "$dir"
  echo "== $kv @ context=$context stride=$stride =="
  "${cmd[@]}"

  local ppl="" tokens=""
  if [[ -n "$py" && -f "$dir/report.json" ]]; then
    ppl="$("$py" -c 'import json,sys; d=json.load(open(sys.argv[1]))["overall"]; print("%.6f" % d["perplexity"])' "$dir/report.json")"
    tokens="$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall"]["scored_tokens"])' "$dir/report.json")"
  fi
  printf '%-10s %-8s %-8s %-12s %s\n' "$kv" "$context" "$stride" "${tokens:-n/a}" "${ppl:-n/a}" >> "$out/summary.tsv"
}

rm -f "$out/summary.tsv"
mkdir -p "$out"

if [[ "$skip_bf16" -eq 0 ]]; then
  run bf16 "$ctx64" "$stride64"
fi
run rk8v4 "$ctx128" "$stride128"
run rk6v4-e8 "$ctx128" "$stride128"
run rk8v4 "$ctx200" "$stride200"
run rk6v4-e8 "$ctx200" "$stride200"

if [[ "$dry_run" -eq 0 ]]; then
  echo
  echo "summary (kv, context, stride, scored_tokens, ppl):"
  cat "$out/summary.tsv"
  echo
  echo "reports: $out/<kv-dtype>.<context>/report.json"
  echo "note: the bf16 row runs at $ctx64 while the quantized rows run at $ctx128/$ctx200;"
  echo "state this depth asymmetry in any report built from these runs."
fi
