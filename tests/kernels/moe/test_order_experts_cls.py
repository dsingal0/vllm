# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoE oracle modular-over-monolithic ordering."""

import pytest

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.base import order_experts_cls

pytestmark = pytest.mark.cpu_test


class _Mono(mk.FusedMoEExpertsMonolithic):
    pass


class _Modular(mk.FusedMoEExpertsModular):
    pass


class _Other(mk.FusedMoEExperts):
    pass


def test_order_experts_cls_default_preserves_order():
    classes = [_Mono, _Modular, _Other]
    assert order_experts_cls(classes, prefer_modular=False) == classes


def test_order_experts_cls_prefer_modular_drops_monolithic():
    """When modular exists, mono is skipped so the oracle can fall through."""
    classes = [_Mono, _Modular, _Other]
    ordered = order_experts_cls(classes, prefer_modular=True)
    assert ordered == [_Modular, _Other]
    assert _Mono not in ordered


def test_order_experts_cls_prefer_modular_keeps_mono_if_no_modular():
    classes = [_Mono, _Other]
    assert order_experts_cls(classes, prefer_modular=True) == classes


def test_order_experts_cls_preserves_relative_order_within_groups():
    class _ModularB(mk.FusedMoEExpertsModular):
        pass

    class _MonoB(mk.FusedMoEExpertsMonolithic):
        pass

    classes = [_Mono, _Modular, _MonoB, _ModularB]
    ordered = order_experts_cls(classes, prefer_modular=True)
    assert ordered == [_Modular, _ModularB]
    assert _Mono not in ordered and _MonoB not in ordered
