# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)


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
