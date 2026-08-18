"""Focused correctness checks for the Ascend GDN override hooks."""

import torch

import megatron.plugin  # noqa: F401 - loads centralized override registrations
from megatron.core.ssm.gated_delta_net import apply_gdn_qk_l2norm
from megatron.plugin.decorators import get_override_method


def test_ascend_gdn_hooks_are_selected():
    expected_module = "megatron.plugin.Ascend.ssm.gated_delta_net"
    l2norm_impl = get_override_method("gated_delta_net.apply_gdn_qk_l2norm")
    recurrence_impl = get_override_method("gated_delta_net.run_gated_delta_rule")
    assert l2norm_impl.__module__ == expected_module
    assert recurrence_impl.__module__ == expected_module


def test_ascend_gdn_l2norm_forward_backward():
    torch.manual_seed(123)
    ref_input = torch.randn(
        2, 128, 8, 128, device="npu", dtype=torch.bfloat16, requires_grad=True
    )
    test_input = ref_input.detach().clone().requires_grad_(True)

    ref_norm = ref_input / torch.norm(
        ref_input, p=2, dim=-1, keepdim=True
    ).clamp(min=1e-6)
    test_norm, _ = apply_gdn_qk_l2norm(test_input, test_input.detach())
    torch.testing.assert_close(test_norm, ref_norm, rtol=2e-2, atol=2e-3)

    output_grad = torch.randn_like(ref_norm)
    ref_norm.backward(output_grad)
    test_norm.backward(output_grad)
    torch.testing.assert_close(test_input.grad, ref_input.grad, rtol=3e-2, atol=3e-3)
