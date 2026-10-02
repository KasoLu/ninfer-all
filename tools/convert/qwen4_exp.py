"""Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration) adapter: the mathematical text config.

The architecture: 48 blocks of Gated DeltaNet and Qwen Sparse Attention (QSA) mixers, each half
read from and written into a four-stream hyper-connection residual, a 512-expert top-10 MoE in
every block, and a hashed n-gram embedding (PLE) injected into the residual before one block.
The converter normalises the checkpoint's config into the text component's config, which
src/models/qwen4_exp/config.cpp parses strictly.
"""

from __future__ import annotations

from .config_fields import _f32, _fixed, _positive, _rope_source

_ARCHITECTURES = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")

# The reference's class default for the n-gram multipliers (absent from config.json).
_NGRAM_SEED = 1234


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected a nonnegative integer")
    return value


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _ARCHITECTURES
    ):
        raise ValueError(f"unsupported Qwen4Exp architecture {architectures!r}")
    raw = source.get("text_config", source)
    _fixed(raw, "model_type", "qwen4_exp_text", "text")
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "mamba_ssm_dtype", "float32", "text")
    _fixed(raw, "output_gate_type", "sigmoid", "text")
    _fixed(raw, "indexer_kv_heads", 1, "text")
    result = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text"}
    for key in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "max_position_embeddings",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "hc_count",
        "hc_lowrank",
        "indexer_n_heads",
        "indexer_head_dim",
        "indexer_compress_ratio",
        "indexer_budget",
        "ple_embed_dim",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "ple_conv_kernel_size",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
    if type(tied) is not bool:
        raise ValueError("text.tie_word_embeddings must be boolean")
    result["tie_word_embeddings"] = tied
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    result["eos_token_id"] = _nonnegative(raw.get("eos_token_id"), "text.eos_token_id")
    if result["eos_token_id"] >= result["vocab_size"]:
        raise ValueError("text.eos_token_id exceeds the vocabulary")
    result["ngram_seed"] = _nonnegative(raw.get("seed", _NGRAM_SEED), "text.seed")

    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("full_attention", "linear_attention") for k in layers)
    ):
        raise ValueError("text.layer_types must describe every block")
    result["layer_types"] = list(layers)

    rope = _rope_source(raw, "text")
    _fixed(rope, "mrope_interleaved", True, "text.rope_parameters")
    factor = _f32(rope.get("partial_rotary_factor", raw.get("partial_rotary_factor")),
                  "partial_rotary_factor")
    sections = rope.get("mrope_section")
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(type(v) is not int or v < 0 for v in sections)
    ):
        raise ValueError("mrope_section must contain three nonnegative integers")
    result["rope_parameters"] = {
        "rope_theta": _f32(rope.get("rope_theta", raw.get("rope_theta")), "rope_theta"),
        "partial_rotary_factor": factor,
        "mrope_section": list(sections),
    }

    # ple_layer_ids is one-indexed in the checkpoint ([2] is the second block); the artifact
    # stores zero-based block indices.
    ple = raw.get("ple_layer_ids")
    if not isinstance(ple, list) or not ple:
        raise ValueError("text.ple_layer_ids must name at least one block")
    blocks = []
    for value in ple:
        index = _positive(value, "text.ple_layer_ids") - 1
        if index >= result["num_hidden_layers"] or layers[index] != "linear_attention":
            raise ValueError("text.ple_layer_ids must name Gated DeltaNet blocks")
        blocks.append(index)
    if blocks != sorted(set(blocks)):
        raise ValueError("text.ple_layer_ids must be increasing")
    result["ple_layers"] = blocks

    if result["num_experts_per_tok"] > result["num_experts"]:
        raise ValueError("selected experts exceed expert count")
    if result["indexer_budget"] % result["indexer_compress_ratio"]:
        raise ValueError("text.indexer_budget must be a whole number of compressed blocks")
    if result["ple_embed_dim"] % ((result["ngram_size"] - 1) * result["heads_per_ngram"]):
        raise ValueError("text.ple_embed_dim must divide among the n-gram heads")
    return result


def mtp_config(source: dict) -> dict:
    raw = source.get("text_config", source)
    _fixed(raw, "mtp_num_hidden_layers", 1, "text")
    _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    mtp = raw.get("mtp")
    if not isinstance(mtp, dict):
        raise ValueError("text.mtp must describe the MTP block")
    _fixed(mtp, "num_hidden_layers", 1, "text.mtp")
    _fixed(mtp, "hybrid", True, "text.mtp")
    _fixed(mtp, "layer_types", ["full_attention"], "text.mtp")
    return {
        "architectures": ["Qwen4ExpMTP"],
        "rope_theta": _f32(mtp.get("rope_theta"), "text.mtp.rope_theta"),
    }
