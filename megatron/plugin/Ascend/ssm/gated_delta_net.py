"""Ascend overrides for Gated Delta Net."""

from __future__ import annotations

from fla_npu.ops.triton import l2norm

from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
from .npu_chunk_gated_delta_rule import npu_chunk_gated_delta_rule


def apply_gdn_qk_l2norm(query, key):
    """Use fla_npu's NPU L2Norm."""
    return l2norm(query.contiguous()), l2norm(key.contiguous())


def run_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    deterministic_mode,
):
    """Use the AscendC recurrence unless deterministic execution was requested."""
    if deterministic_mode:
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

    return npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
