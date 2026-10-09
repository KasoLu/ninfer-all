#!/usr/bin/env bash
# run_long_context_ppl.sh — one-command driver for the rk6v4-e8 long-context PPL acceptance run.
#
# Does, in order:
#   1. the ctest gate: i6 round-trip hard gate + kv_cache_append tests (must be green)
#   2. corpus preparation: uses an existing Journey_to_the_West.txt if found, otherwise
#      downloads it from modelscope (AI-ModelScope/Needle-in-a-Haystack-Corpus) into corpora/
#   3. the depth matrix from long_context_ppl.sh: bf16 @ 64K, rk8v4 + rk6v4-e8 @ 128K / 200K
#
# The build is the operator's job — build these two targets first (the test bundle and the
# perplexity app pull in the core, so this compiles essentially everything):
#   cmake --build build-ninja --target ninfer-perplexity --target ninfer_tests -j
#
# usage: tools/run_long_context_ppl.sh <model.ninfer> [options]
#   --text FILE            long UTF-8 file (default Journey_to_the_West.txt)
#   --out DIR              report root (default profiles/perplexity/long_ctx)
#   --skip-bf16            omit the bf16 64K row
#   --skip-gate            skip the ctest gate (not recommended)
#   --no-download          fail instead of downloading a missing corpus file
#   --no-install           fail instead of pip-installing modelscope into the system python
#   --dry-run              print every step without executing anything
#   --contextNNN N / --strideNNN N   per-depth overrides (64/128/200), passed through

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

model="" text="" text_given=0 out="" gate=1 download=1 install=1 dry_run=0 skip_bf16=0
passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   dry_run=1 ;;
    --skip-bf16) skip_bf16=1 ;;
    --skip-gate) gate=0 ;;
    --no-download) download=0 ;;
    --no-install)  install=0 ;;
    --text)      text="$2"; text_given=1; shift ;;
    --out)       out="$2"; shift ;;
    --context64|--stride64|--context128|--stride128|--context200|--stride200)
      passthrough+=("$1" "$2"); shift ;;
    --help|-h)   sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)          echo "unknown option: $1" >&2; exit 1 ;;
    *)           if [[ -z "$model" ]]; then model="$1"
                 else echo "unexpected argument: $1" >&2; exit 1
                 fi ;;
  esac
  shift
done
[[ -n "$model" ]] || { echo "usage: tools/run_long_context_ppl.sh <model.ninfer> [options]" >&2; exit 1; }
[[ "$text_given" -eq 1 ]] || text="Journey_to_the_West.txt"

announce() { if [[ "$dry_run" -eq 1 ]]; then echo "[dry-run] $*"; else echo "== $* =="; fi; }

# -- build tree and binary -----------------------------------------------------------
build_tree=""
for tree in build-ninja build; do
  [[ -f "$root/$tree/CTestTestfile.cmake" ]] && build_tree="$root/$tree" && break
done
binary=""
for tree in build-ninja build; do
  for candidate in "$root/$tree/apps/ninfer-perplexity" "$root/$tree/apps/ninfer-perplexity.exe"; do
    [[ -f "$candidate" ]] && binary="$candidate" && break
  done
  [[ -n "$binary" ]] && break
done
if [[ "$dry_run" -eq 1 ]]; then
  announce "binary: ${binary:-<will be required>}, build tree: ${build_tree:-<will be required>}"
else
  [[ -n "$binary" ]] || { echo "error: ninfer-perplexity not found — build it first:" >&2
    echo "  cmake --build build-ninja --target ninfer-perplexity --target ninfer_tests -j" >&2
    exit 1; }
  [[ -n "$build_tree" ]] || { echo "error: no CMake build tree (build-ninja/ or build/) — configure first" >&2; exit 1; }
  echo "binary: $binary"
fi

# -- 1. ctest gate -------------------------------------------------------------------
if [[ "$gate" -eq 1 ]]; then
  if [[ "$dry_run" -eq 1 ]]; then
    announce "gate: ctest --test-dir ${build_tree:-<build tree>} -R 'i6_roundtrip|kv_cache_append' --output-on-failure"
  else
    announce "gate: i6 round-trip + kv_cache_append"
    ctest --test-dir "$build_tree" -R 'i6_roundtrip|kv_cache_append' --output-on-failure
  fi
fi

# -- 2. corpus preparation ------------------------------------------------------------
text_path=""
for candidate in "$text" "corpora/$text" "eval/corpora/$text"; do
  [[ -f "$candidate" ]] && text_path="$candidate" && break
done
if [[ -n "$text_path" ]]; then
  announce "corpus: using existing $text_path"
elif [[ "$dry_run" -eq 1 ]]; then
  text_path="$text"
  announce "corpus: would download $text (modelscope AI-ModelScope/Needle-in-a-Haystack-Corpus) into corpora/$text"
elif [[ "$download" -eq 0 ]]; then
  echo "error: $text not found (searched ., corpora/, eval/corpora/) and --no-download is set" >&2
  exit 1
else
  announce "corpus: downloading $text from modelscope"
  py=""
  for candidate in eval/.venv/bin/python python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import modelscope' >/dev/null 2>&1; then
      py="$candidate"; break
    fi
  done
  if [[ -z "$py" && "$install" -eq 1 ]]; then
    for candidate in python3 python; do
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -m pip --version >/dev/null 2>&1; then
        announce "corpus: installing modelscope into $candidate"
        if "$candidate" -m pip install --quiet modelscope >/dev/null 2>&1 \
          && "$candidate" -c 'import modelscope' >/dev/null 2>&1; then
          py="$candidate"
        fi
        break
      fi
    done
  fi
  if [[ -z "$py" ]]; then
    echo "error: no python with the modelscope module (tried eval/.venv/bin/python, python3, python${install:+, attempted pip install})." >&2
    echo "rerun without --no-install, or download the file manually and put it in corpora/:" >&2
    echo "  python -c \"from modelscope import dataset_snapshot_download; dataset_snapshot_download('AI-ModelScope/Needle-in-a-Haystack-Corpus', allow_file_pattern=['$text'])\"" >&2
    exit 1
  fi
  snapshot="$("$py" -c "from modelscope import dataset_snapshot_download; print(dataset_snapshot_download('AI-ModelScope/Needle-in-a-Haystack-Corpus', allow_file_pattern=['$text']))")"
  src="$snapshot/$text"
  [[ -f "$src" ]] || { echo "error: download succeeded but $src is missing" >&2; exit 1; }
  mkdir -p corpora
  cp "$src" "corpora/$text"
  text_path="corpora/$text"
  echo "corpus: saved to $text_path"
fi

extra=()
[[ "$skip_bf16" -eq 1 ]] && extra+=("--skip-bf16")
[[ -n "$out" ]] && extra+=("--out" "$out")
if [[ "$dry_run" -eq 1 ]]; then
  announce "matrix: tools/long_context_ppl.sh $model $text_path --dry-run ${extra[*]:-}"
else
  bash tools/long_context_ppl.sh "$model" "$text_path" "${extra[@]+"${extra[@]}"}" ${passthrough[@]+"${passthrough[@]}"}
fi
