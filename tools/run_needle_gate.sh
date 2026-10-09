#!/usr/bin/env bash
# run_needle_gate.sh -- needle retrieval gate for the rk6v4-e8 acceptance
# (RK6V4E8.md sections 7.3/7.4): the Engine must retrieve a buried needle at
# the 200K context tier.
#
# Usage:
#   tools/run_needle_gate.sh MODEL.ninfer [options]
#
# Options:
#   --max-context N    Engine context for every run (default 200000)
#   --single-tokens N  haystack token target for the single-needle probe
#                      (default: the --max-context value; the generator keeps
#                      a 10% safety margin, so 200000 -> ~180K est tokens)
#   --formats LIST     space-separated kv-dtype list (default "rk6v4-e8 rk8v4")
#   --gates LIST       subset of "single code five" (default "single code")
#   --probes-dir DIR   probe output directory (default corpora/needles)
#   --out DIR          result directory (default profiles/needle_gate)
#   --skip-gate        skip the i6_roundtrip/kv_cache_append ctest gate
#   --dry-run          print every step without executing anything
#
# Flow:
#   0. ctest -R 'i6_roundtrip|kv_cache_append' --output-on-failure
#      (the byte-exact round-trip gate; a byte-shifted i6 layout corrupts 12.5%
#      of key dimensions and short samples do not show it)
#   1. tools/bench/make_needle_probes.py generates the probes + expected answers
#      (deterministic seed, stdlib only, no network)
#   2. for each kv-dtype and each selected gate: build/apps/ninfer with
#      --messages <probe> --greedy, output captured to <out>/<fmt>.<gate>.log
#   3. grade every log against expected_answers.json (substring / order checks)
#      and write <out>/verdicts.tsv
#
# Exit code 0 iff every selected gate passed in every selected format.
#
# Note on bf16: bf16 KV at 200K context is ~12 GiB and does not fit next to a
# 27B model on a 24 GB card, so the default formats exclude it. Adding bf16
# to --formats is only meaningful with a shallow --max-context; the depth
# asymmetry must then be stated in the report (same convention as the
# long-context PPL driver).

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

announce() {
  if [[ "$dry_run" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "$@"
  fi
}

model="" ctx=200000 single_tokens="" formats="rk6v4-e8 rk8v4" gates="single code"
probes="corpora/needles" out="profiles/needle_gate" gate_on=1 dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-context)   ctx="$2"; shift 2 ;;
    --single-tokens) single_tokens="$2"; shift 2 ;;
    --formats)       formats="$2"; shift 2 ;;
    --gates)         gates="$2"; shift 2 ;;
    --probes-dir)    probes="$2"; shift 2 ;;
    --out)           out="$2"; shift 2 ;;
    --skip-gate)     gate_on=0; shift ;;
    --dry-run)       dry_run=1; shift ;;
    -h|--help)       sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)              echo "unknown option: $1" >&2; exit 2 ;;
    *)               model="$1"; shift ;;
  esac
done

[[ -n "$model" ]] || { echo "usage: $0 MODEL.ninfer [options]   (see --help)" >&2; exit 2; }
[[ -f "$model" ]] || { echo "error: model artifact not found: $model" >&2; exit 1; }
[[ -n "$single_tokens" ]] || single_tokens="$ctx"

for g in $gates; do
  case "$g" in single|code|five) ;; *) echo "unknown gate: $g (single|code|five)" >&2; exit 2 ;; esac
done
for f in $formats; do
  case "$f" in bf16|rk8v4|rk6v4-e8|rk4v4-e8|rk2v4-e8|rk4v4|i8v4) ;; *)
    echo "unknown kv-dtype: $f" >&2; exit 2 ;; esac
done

py=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -V >/dev/null 2>&1; then
    py="$candidate"; break
  fi
done
[[ -n "$py" ]] || { echo "error: no working python3/python found" >&2; exit 1; }
bin="$root/build/apps/ninfer"
[[ -f "$bin" ]] || { echo "error: $bin not built (cmake --build build --target ninfer)" >&2; exit 1; }

echo "needle gate: model=$model  ctx=$ctx  single_tokens=$single_tokens  formats: $formats  gates: $gates"
if [[ "$ctx" -gt 80000 ]] && [[ " $formats " == *" bf16 "* ]]; then
  echo "note: bf16 KV at ctx=$ctx is ~$((ctx * 63928 / 1073741824)) GiB (63,928 B/token across all heads/layers) and may not fit on a 24 GB card; the bf16 row, if it runs, is a shallow-equivalent reference only"
fi

# --- 0. round-trip gate ---------------------------------------------------
if [[ "$gate_on" -eq 1 ]]; then
  announce "gate: ctest --test-dir $root/build -R 'i6_roundtrip|kv_cache_append' --output-on-failure"
  if [[ "$dry_run" -eq 0 ]]; then
    ctest --test-dir "$root/build" -R 'i6_roundtrip|kv_cache_append' --output-on-failure
  fi
else
  announce "gate: skipped (--skip-gate)"
fi

# --- 1. probes --------------------------------------------------------------
announce "probes: $py tools/bench/make_needle_probes.py --out $probes --single-tokens $single_tokens --cpl-text 4.75"
if [[ "$dry_run" -eq 0 ]]; then
  "$py" tools/bench/make_needle_probes.py --out "$probes" --single-tokens "$single_tokens" --cpl-text 4.75
  mkdir -p "$out"
fi

# --- 2. runs -----------------------------------------------------------------
for fmt in $formats; do
  for g in $gates; do
    case "$g" in
      five) max_new=128 ;;
      *)    max_new=32 ;;
    esac
    probe="$probes/needle_$g.json"
    log="$out/$fmt.$g.log"
    announce "run: $bin $model --kv-dtype $fmt --max-context $ctx --no-thinking --messages $probe --max-new $max_new --greedy   (-> $log)"
    if [[ "$dry_run" -eq 0 ]]; then
      "$bin" "$model" --kv-dtype "$fmt" --max-context "$ctx" --no-thinking \
        --messages "$probe" --max-new "$max_new" --greedy > "$log" 2>&1
    fi
  done
done

# --- 3. grading ---------------------------------------------------------------
announce "grade: $py (substring/order checks against $probes/expected_answers.json -> $out/verdicts.tsv)"
if [[ "$dry_run" -eq 0 ]]; then
  "$py" - "$probes" "$out" "$formats" "$gates" <<'PY'
import json, sys
probes_dir, out_dir, formats, gates = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
expected = json.load(open(f"{probes_dir}/expected_answers.json"))
phrases = expected["passphrases"]
canary = expected["canary"]
rows = []
failed = 0
import os
for fmt in formats.split():
    for g in gates.split():
        log = f"{out_dir}/{fmt}.{g}.log"
        if not os.path.exists(log):
            rows.append((fmt, g, "MISSING", 0))
            failed += 1
            continue
        text = open(log, encoding="utf-8", errors="replace").read()
        if g == "single":
            ok = phrases[0] in text
        elif g == "five":
            pos = [text.find(p) for p in phrases[:5]]
            ok = all(p >= 0 for p in pos) and pos == sorted(pos)
        elif g == "code":
            ok = str(canary["kCanaryValue"]) in text and str(canary["computeCanary"]) in text
        else:
            ok = False
        rows.append((fmt, g, "PASS" if ok else "FAIL",
                     expected["probes"][f"needle_{g}"]["estimated_tokens"]))
        if not ok:
            failed += 1
with open(f"{out_dir}/verdicts.tsv", "w") as f:
    f.write("kv-dtype\tgate\tresult\testimated_tokens\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print("\n".join(f"  {r[0]}\t{r[1]}\t{r[2]}" for r in rows))
a = [r for r in rows if r[1] == "single"]
c = [r for r in rows if r[1] == "code"]
print(f"gate A (single-needle, ctx tier): {'PASS' if a and all(r[2]=='PASS' for r in a) else 'FAIL'}")
print(f"gate C (code-detail):             {'PASS' if c and all(r[2]=='PASS' for r in c) else 'FAIL'}")
sys.exit(1 if failed else 0)
PY
fi
