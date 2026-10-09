#!/usr/bin/env python3
"""Deterministic needle-in-a-haystack probe generator for the retrieval gates.

Produces three OpenAI-style messages JSON files for the ``ninfer`` CLI
(``--messages``), matching the retrieval-gate profile in RK6V4E8.md section 7.3:

  * needle_single.json  -- single passphrase needle at 50% depth of a haystack
                          sized by --single-tokens (default ~260K tokens, which
                          fits --max-context 262144)
  * needle_five.json    -- five passphrase needles at depths
                          0.05/0.25/0.50/0.75/0.95 of ~118K tokens
                          (fits --max-context 122880)
  * needle_code.json    -- exact-code-detail probe: one distinctive C++ function
                          (kCanaryValue) buried in ~168K tokens of C++ filler
                          (fits --max-context 172032)

Every probe is one long user message (haystack + trailing question), which
the CLI tokenizes with the model's own tokenizer. Token counts are
*estimated* (chars / chars-per-token); the --safety factor keeps the
estimate under the context limit with headroom for tokenizer-model
estimation error. Depths are expressed as fractions of the haystack, so the
gate semantics hold regardless of the exact tokenization.

Output: <out>/<probe>.json for each probe plus <out>/expected_answers.json
(passphrases / canary values / generation metadata) so results can be
graded mechanically.

Usage:
  python3 tools/bench/make_needle_probes.py --out /tmp/needles
  # size the single-needle haystack to a 200K context:
  python3 tools/bench/make_needle_probes.py --out /tmp/needles --single-tokens 200000
  # smoke test at 1% size:
  python3 tools/bench/make_needle_probes.py --out /tmp/needles_smoke --scale 0.01

Stdlib only (random, json, argparse, pathlib) so it runs in any dev container
as-is.
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed, distinctive needles (human-checkable, seed-independent on purpose).
# ---------------------------------------------------------------------------

PASSPHRASES = [
    "XK7Q-93LM-TW4Z-PQ8R",
    "VB2N-61HD-RY7C-MX5K",
    "ZT8F-34WS-QE9J-LU1D",
    "CW5G-72XP-NO3B-TA6H",
    "YR4M-18KC-ZD6V-SB2Q",
]

CANARY_VALUE = 402182
CANARY_RESULT = CANARY_VALUE * 7 + 13  # 2815287

CODE_NEEDLE_BLOCK = f"""\
namespace needle_probe {{
// Retrieval-gate canary. This constant appears exactly once in the file above.
constexpr std::uint32_t kCanaryValue = {CANARY_VALUE}u;

std::uint32_t computeCanary() {{
    return kCanaryValue * 7u + 13u;  // exact expected result: {CANARY_RESULT}
}}
}}  // namespace needle_probe
"""

# ---------------------------------------------------------------------------
# Filler banks (seeded generators; repetition inside the haystack is fine).
# ---------------------------------------------------------------------------

_NOUNS = [
    "harbor", "meadow", "lantern", "bridge", "orchard", "canyon", "mill",
    "lighthouse", "vineyard", "garrison", "foundry", "quarry", "beacon",
    "terrace", "granary", "passage", "outpost", "belltower", "dockyard",
    "watchtower", "airfield", "reservoir", "barracks", "signal tower",
    "greenhouse", "windmill", "cannery", "staircase", "waystation", "embankment",
]
_VERBS = [
    "traced", "mirrored", "outpaced", "anchored", "ruffled", "polished",
    "measured", "shadowed", "braced", "tempered", "sorted", "folded",
    "charted", "kindled", "harvested", "catalogued", "rattled", "glistened",
    "settled", "drifted", "clustered", "trembled", "blossomed", "steered",
]
_ADJ = [
    "quiet", "amber", "weathered", "silver", "distant", "gilded", "patient",
    "rustic", "hollow", "luminous", "faded", "sturdy", "gentle", "pale",
    "wandering", "seasoned", "copper", "murmuring", "slanting", "brisk",
    "sunlit", "mottled", "vigilant", "tranquil",
]
_PLACES = [
    "the north wall", "the river bend", "the old gate", "the harbor mouth",
    "the limestone ridge", "the market square", "the eastern ridge",
    "the mill race", "the town clock", "the grain market", "the signal mast",
    "the quarry face", "the breakwater", "the chapel yard", "the watch road",
    "the ferry landing", "the orchard lane", "the customs shed",
    "the bell tower", "the stone arch",
]

_CODE_NOUN = [
    "cache", "tile", "page", "lane", "block", "head", "slot", "norm",
    "proj", "gate", "bias", "chunk", "window", "score", "draft", "spec",
    "token", "embed", "mask", "scale", "step", "pack", "shift", "blend",
    "route", "stamp", "probe", "weave", "zero", "fold",
]
_CODE_VERB = [
    "make", "build", "apply", "compute", "update", "merge", "split", "fetch",
    "store", "load", "copy", "fill", "read", "write", "swap", "clamp",
    "scan", "emit", "gather", "scatter", "slide", "trim", "bind", "mix",
    "yield",
]


def _svo_line(rng: random.Random) -> str:
    return (
        f"The {rng.choice(_ADJ)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} "
        f"the {rng.choice(_ADJ)} {rng.choice(_NOUNS)} beside {rng.choice(_PLACES)}."
    )


def _code_line(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.18:
        return f"// step {rng.randrange(2, 4096)}: pass the {rng.choice(_CODE_NOUN)} {rng.choice(_CODE_VERB)} to the lane"
    if kind < 0.30:
        return (
            f"static int {rng.choice(_CODE_VERB)}{rng.choice(_CODE_NOUN)}(int a, int b) {{ "
            f"return a * {rng.randrange(2, 64)} + b; }}"
        )
    if kind < 0.42:
        return (
            f"struct {rng.choice(_CODE_VERB).capitalize()}{rng.choice(_CODE_NOUN).capitalize()} {{ "
            f"int head_{rng.randrange(1, 16)}; int lane_{rng.randrange(1, 16)}; int scale_{rng.randrange(2, 256)}; }};"
        )
    stmts = [
        f"int t{rng.randrange(1, 8)} = {rng.randrange(1, 512)} * {rng.randrange(1, 512)};",
        f"const int base = {rng.randrange(64, 4096)} >> {rng.randrange(1, 5)};",
        f"for (int {rng.choice('xyzt')} = 0; {rng.choice('xyzt')} < {rng.randrange(8, 64)}; ++{rng.choice('xyzt')}) {{",
        f"  {rng.choice(_CODE_VERB)}{rng.choice(_CODE_NOUN)}({rng.randrange(0, 64)}, {rng.randrange(0, 64)});",
        f"  sum += {rng.choice('xyzt')} * {rng.randrange(2, 32)};",
        f"  if ({rng.choice('xyzt')} & {rng.randrange(1, 32)}) continue;",
        f"return {rng.choice('xyzt')} + {rng.randrange(1, 128)};",
        "}",
    ]
    return rng.choice(stmts)


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def _generate_haystack(rng, target_chars: int, line_fn, needles=None) -> tuple[str, dict]:
    """Generate filler lines until target_chars is reached; splice in needles.

    needles: sorted list of (depth_fraction, line). Returned: (haystack_text,
    info about needle line positions).
    """
    lines: list[str] = []
    chars = 0
    # Average ~60 chars per line for SVO / ~70 for code; estimate generously
    # and stop when the running total clears the target.
    while chars < target_chars:
        line = line_fn(rng)
        lines.append(line)
        chars += len(line) + 1

    info = {"filler_lines": len(lines), "filler_chars": chars}
    if needles:
        total = len(lines)
        last_pos = -1
        for depth, line in sorted(needles, key=lambda n: n[0]):
            pos = max(int(depth * total), last_pos + 5)
            pos = min(pos, total - 1)
            lines.insert(pos, line)
            info.setdefault("needles", []).append({"depth": depth, "line": line})
            last_pos = pos

    return "\n".join(lines) + "\n", info


def _probe_messages(haystack: str, question: str) -> list[dict]:
    return [{"role": "user", "content": haystack + question}]


def build_probe_single(rng, est_tokens: int, cpl: float, depth: float) -> dict:
    est_chars = int(est_tokens * cpl)
    hay, info = _generate_haystack(
        rng, est_chars, lambda r: _svo_line(r), [(depth, f"The secret passphrase is {PASSPHRASES[0]}.")]
    )
    question = (
        "\nSomewhere in the long text above there is exactly one line stating "
        "a secret passphrase made of four letter/digit groups separated by hyphens. "
        "What is that exact passphrase? Answer with the passphrase only, nothing else.\n"
    )
    return {
        "name": "needle_single",
        "ctx": 262144,
        "messages": _probe_messages(hay, question),
        "expected": PASSPHRASES[0],
        "estimated_tokens": int((len(hay) + len(question)) / cpl),
        "max_new": 32,
        "info": info,
    }


def build_probe_five(rng, est_tokens: int, cpl: float, depths) -> dict:
    est_chars = int(est_tokens * cpl)
    needles = [(d, f"The hidden passphrase #{i + 1} is {PASSPHRASES[i]}.")
               for i, d in enumerate(depths)]
    hay, info = _generate_haystack(rng, est_chars, lambda r: _svo_line(r), needles)
    question = (
        "\nExactly five lines in the long text above each state a hidden passphrase "
        "(four letter/digit groups separated by hyphens), numbered #1 through #5 in "
        "order of appearance. List all five passphrases in that order, one per line, "
        "exactly as written. No other text.\n"
    )
    return {
        "name": "needle_five",
        "ctx": 122880,
        "messages": _probe_messages(hay, question),
        "expected": "\n".join(PASSPHRASES[:5]),
        "estimated_tokens": int((len(hay) + len(question)) / cpl),
        "max_new": 128,
        "info": info,
    }


def build_probe_code(rng, est_tokens: int, cpl: float, depth: float) -> dict:
    est_chars = int(est_tokens * cpl)
    hay, info = _generate_haystack(
        rng, est_chars, lambda r: _code_line(r), [(depth, CODE_NEEDLE_BLOCK)]
    )
    question = (
        "\nIn the C++ source above there is a namespace needle_probe whose function "
        "computeCanary() multiplies the constant kCanaryValue by 7 and adds 13. "
        "What is the exact value of kCanaryValue, and what exact integer does "
        "computeCanary() return? Answer with the two decimal numbers only.\n"
    )
    return {
        "name": "needle_code",
        "ctx": 172032,
        "messages": _probe_messages(hay, question),
        "expected": f"{CANARY_VALUE} and {CANARY_RESULT}",
        "estimated_tokens": int((len(hay) + len(question)) / cpl),
        "max_new": 32,
        "info": info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="needle_probes", help="output directory")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink all token targets by this factor (smoke tests)")
    ap.add_argument("--safety", type=float, default=0.90,
                    help="multiply target tokens by this to leave context headroom")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--cpl-text", type=float, default=2.5,
                    help="conservative chars-per-token estimate for prose filler")
    ap.add_argument("--cpl-code", type=float, default=2.2,
                    help="conservative chars-per-token estimate for code filler")
    ap.add_argument("--single-tokens", type=int, default=260000,
                    help="haystack token target for the single-needle probe "
                         "(keep under the Engine --max-context of the gate run)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    budget = lambda tokens: int(tokens * args.scale * args.safety)
    probes = [
        build_probe_single(rng, budget(args.single_tokens), args.cpl_text, 0.50),
        build_probe_five(rng, budget(118000), args.cpl_text,
                         [0.05, 0.25, 0.50, 0.75, 0.95]),
        build_probe_code(rng, budget(168000), args.cpl_code, 0.50),
    ]

    expected = {
        "seed": args.seed, "scale": args.scale, "safety": args.safety,
        "cpl_text": args.cpl_text, "cpl_code": args.cpl_code,
        "single_tokens": args.single_tokens,
        "passphrases": list(PASSPHRASES),
        "canary": {"kCanaryValue": CANARY_VALUE, "computeCanary": CANARY_RESULT},
        "probes": {},
    }
    for probe in probes:
        path = out / f"{probe['name']}.json"
        path.write_text(json.dumps(probe["messages"], ensure_ascii=False) + "\n",
                        encoding="utf-8")
        expected["probes"][probe["name"]] = {
            "file": str(path),
            "expected_answer": probe["expected"],
            "estimated_tokens": probe["estimated_tokens"],
            "ctx": probe["ctx"],
            "max_new": probe["max_new"],
            "filler_lines": probe["info"]["filler_lines"],
            "needles": [n["depth"] for n in probe["info"].get("needles", [])],
        }
        print(f"wrote {path}  est_tokens={probe['estimated_tokens']:>8}  "
              f"ctx={probe['ctx']}  lines={probe['info']['filler_lines']}")

    meta = out / "expected_answers.json"
    meta.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta}  (expected answers + generation metadata)")
    print("\ngate summary (grade against expected_answers.json):")
    for probe in probes:
        print(f"  {probe['name']}: ctx={probe['ctx']}  max_new={probe['max_new']}")


if __name__ == "__main__":
    main()
