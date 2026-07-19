# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def test_parent_module_skips_descendants_only():
    ignored = ["model.layers.0.self_attn", "model.layers.0.mlp.gate"]

    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=ignored,
    )
    assert not is_layer_skipped(
        prefix="model.layers.0.mlp.gate_up_proj",
        ignored_layers=ignored,
    )


def test_parent_module_skips_all_attention_children():
    # GLM-5.2-FP8blk-bf16attn lists the parent ``self_attn`` in
    # ``ignored_layers``; every child linear (q_proj, o_proj, the fused
    # qkv_proj, kv_a_proj_with_mqa, ...) must be skipped.
    ignored = ["model.layers.0.self_attn"]
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.q_proj",
        ignored_layers=ignored,
    )
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.kv_a_proj_with_mqa",
        ignored_layers=ignored,
    )
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=ignored,
    )
    # Fused qkv_proj under the parent is also skipped via the fused-name
    # parent-prefix match (no per-shard expansion needed).
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )


def test_parent_module_does_not_skip_sibling_layer():
    ignored = ["model.layers.0.self_attn"]
    assert not is_layer_skipped(
        prefix="model.layers.1.self_attn.q_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )


def test_parent_module_respects_dotted_boundary():
    # ``self_attn`` must NOT match ``self_attn_v`` (different module).
    ignored = ["model.layers.0.self_attn"]
    assert not is_layer_skipped(
        prefix="model.layers.0.self_attn_v.q_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )


def test_parent_module_does_not_skip_experts():
    # Experts are not under ``self_attn``; listing ``self_attn`` in
    # ``ignored_layers`` must not bleed into the MoE experts path.
    ignored = ["model.layers.0.self_attn"]
    assert not is_layer_skipped(
        prefix="model.layers.0.mlp.experts.0.gate_up_proj",
        ignored_layers=ignored,
    )
