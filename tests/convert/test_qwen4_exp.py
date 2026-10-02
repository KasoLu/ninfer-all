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


def _tiny_config():
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": 8,
        "vocab_size": 12,
        "num_hidden_layers": 2,
        "max_position_embeddings": 64,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5,
                            "mrope_section": [1, 0, 0], "mrope_interleaved": True},
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 4,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 2,
        "hc_lowrank": 3,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_compress_ratio": 4,
        "indexer_budget": 8,
        "ple_layer_ids": [1],
        "ple_embed_dim": 8,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 100,
        "make_ngram_vocab_size_divisible_by": 8,
        "eos_token_id": 11,
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
    }
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": text}


def test_build_model_maps_the_text_component(tmp_path):
    import torch
    from safetensors.torch import save_file

    from tools.convert.qwen4_exp import build_model
    from tools.convert.sources.safetensors import SafetensorsSource

    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_tiny_config()))
    index_qk = torch.arange(12 * 8).float().reshape(12, 8)
    conv = torch.arange(16 * 4).float().reshape(16, 1, 4)
    save_file({
        "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight": index_qk,
        "model.language_model.layers.0.ple.conv1d.weight": conv,
    }, root / "model.safetensors")
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (root / name).write_text(json.dumps(value))
    (root / "chat_template.jinja").write_text("{{ messages }}")
    with SafetensorsSource(root) as source:
        model = build_model(source)
        names = set(model.parameters)
        assert {"text/final_mixer/norm", "text/final_mixer/up"} <= names
        assert "text/final_mixer/inject" not in names
        assert {"text/layers/0/attn_hc/inject", "text/layers/1/mlp_hc/down"} <= names
        assert {"text/layers/0/gdn/query", "text/layers/1/attention/gate",
                "text/layers/1/indexer/key", "text/layers/0/ple/convolution",
                "text/layers/1/moe/experts/2/down"} <= names
        assert not any(n.startswith("text/layers/1/ple/") for n in names)
        query = model.parameters["text/layers/1/indexer/query"]
        key = model.parameters["text/layers/1/indexer/key"]
        assert query.shape == (8, 8) and key.shape == (4, 8)
        assert torch.equal(query.source.values().reshape(8, 8), index_qk[:8])
        assert torch.equal(key.source.values().reshape(4, 8), index_qk[8:])
        convolution = model.parameters["text/layers/0/ple/convolution"]
        assert convolution.shape == (16, 4)
        assert torch.equal(convolution.source.values().reshape(16, 4), conv[:, 0, :])
