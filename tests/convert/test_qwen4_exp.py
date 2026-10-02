from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.convert.qwen4_exp import mtp_config, text_config

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "qwen4_exp"


def _checkpoint():
    return json.loads((_FIXTURES / "config.json").read_text())


def test_text_config_matches_the_artifact_fixture():
    # The C++ parser's test reads the same fixture, so the two stay in step.
    expected = json.loads((_FIXTURES / "text_config.json").read_text())
    assert text_config(_checkpoint()) == expected


def test_ple_layers_are_zero_based_gdn_blocks():
    config = text_config(_checkpoint())
    assert config["ple_layers"] == [1]
    assert config["layer_types"][1] == "linear_attention"
    assert config["ngram_seed"] == 1234


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.update(output_gate_type="silu"), "output_gate_type"),
        (lambda t: t.update(ple_layer_ids=[4]), "Gated DeltaNet"),
        (lambda t: t.update(indexer_kv_heads=2), "indexer_kv_heads"),
        (lambda t: t.update(indexer_budget=2047), "compressed blocks"),
        (lambda t: t.update(eos_token_id=248320), "vocabulary"),
        (lambda t: t.update(num_experts_per_tok=513), "expert count"),
        (lambda t: t["layer_types"].pop(), "every block"),
    ],
)
def test_text_config_refuses(edit, message):
    source = deepcopy(_checkpoint())
    edit(source["text_config"])
    with pytest.raises(ValueError, match=message):
        text_config(source)


def test_mtp_config():
    assert mtp_config(_checkpoint()) == {
        "architectures": ["Qwen4ExpMTP"],
        "rope_theta": 10000000.0,
    }
