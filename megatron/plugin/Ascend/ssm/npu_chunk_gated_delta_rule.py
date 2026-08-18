# Copyright (c) 2025, BAAI. All rights reserved.
#
# NPU AscendC kernel integration for Gated Delta Net (GDN).
# Replaces the fla triton/CUDA path and the slow PyTorch fallback on Ascend NPU.
#
# Kernel source: flash-linear-attention-npu/torch_custom/fla_npu
# Op signatures from: npu_custom.yaml
#
# Forward pipeline:
#   chunk_local_cumsum -> chunk_scaled_dot_kkt_fwd -> solve_tril ->
#   recompute_w_u_fwd -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o
#
# Backward pipeline:
#   recompute_w_u_fwd (recompute) -> chunk_gated_delta_rule_fwd_h (recompute) ->
#   chunk_bwd_dv_local -> chunk_gated_delta_rule_bwd_dhu -> chunk_bwd_dqkwg ->
#   prepare_wy_repr_bwd_da -> prepare_wy_repr_bwd_full -> chunk_local_cumsum(reverse)

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Lazy-loaded triton-on-NPU ops
_NPU_OPS_AVAILABLE: Optional[bool] = None
_chunk_local_cumsum = None
_chunk_scaled_dot_kkt_fwd = None
_solve_tril = None
_l2norm = None


# ============================================================
# Pure PyTorch implementations of triton utility ops.
# These replace fla's triton kernels which fail to compile on NPU.
# They are lightweight relative to the AscendC compute kernels.
# ============================================================

def _torch_chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = True,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch chunk_local_cumsum: cumsum within each chunk.
    
    Input g: [B, H, T] (head_first=True) or [B, T, H] (head_first=False)
    Output: same shape, cumsum within each chunk of size chunk_size.
    """
    if output_dtype is None:
        output_dtype = g.dtype

    if not head_first:
        g = g.transpose(1, 2)  # -> [B, H, T]

    B, H, T = g.shape
    # Pad T to multiple of chunk_size
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        g = torch.nn.functional.pad(g, (0, pad))

    # Reshape to chunks: [B, H, NT, C]
    g_chunks = g.reshape(B, H, -1, chunk_size).to(output_dtype)

    if reverse:
        g_chunks = g_chunks.flip(-1)

    g_cumsum = g_chunks.cumsum(-1)

    if reverse:
        g_cumsum = g_cumsum.flip(-1)

    # Reshape back and trim
    result = g_cumsum.reshape(B, H, -1)[:, :, :T]

    if not head_first:
        result = result.transpose(1, 2)

    return result.contiguous()


def _torch_chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch chunk_scaled_dot_kkt: compute A = tril(k @ k^T * beta * exp(g_i - g_j)).
    
    Input:
        k: [B, H, T, K]
        g: [B, H, T]  (after cumsum)
        beta: [B, H, T]
    Output:
        A: [B, H, T, chunk_size] — strictly lower-triangular within each chunk
           (stored as a flattened chunk dimension: T = NT * C, so shape [B, H, NT*C, C])
           Actually fla stores it as [B, H, T, C] where T encodes chunk position.
           
    More precisely fla returns A with shape [B, H, NT, C, C] then views as [B, H, T, C].
    Let's follow fla's convention.
    """
    B, H, T, K = k.shape
    C = chunk_size
    pad = (C - T % C) % C
    if pad > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        g = torch.nn.functional.pad(g, (0, pad))
        beta = torch.nn.functional.pad(beta, (0, pad))
    
    T_padded = k.shape[2]
    NT = T_padded // C
    
    # Reshape into chunks: [B, H, NT, C, K] and [B, H, NT, C]
    k_chunks = k.reshape(B, H, NT, C, K).to(output_dtype)
    g_chunks = g.reshape(B, H, NT, C).to(output_dtype)
    beta_chunks = beta.reshape(B, H, NT, C).to(output_dtype)
    
    # fla kernel: kb = k * beta[:, None]; A = kb @ kb^T * exp(g_i - g_j)
    # i.e. A[i,j] = (k[i]*beta[i]) @ (k[j]*beta[j]) * exp(g[i] - g[j])
    kb = k_chunks * beta_chunks.unsqueeze(-1)  # [B, H, NT, C, K]
    kkt = torch.matmul(kb, kb.transpose(-1, -2))  # [B, H, NT, C, C]
    
    # Gate differences: exp(g[i] - g[j])
    g_diff = g_chunks.unsqueeze(-1) - g_chunks.unsqueeze(-2)  # [B, H, NT, C, C]
    decay = torch.exp(g_diff)
    
    A = kkt * decay  # [B, H, NT, C, C]
    
    # Strictly lower triangular
    mask = torch.tril(torch.ones(C, C, device=k.device, dtype=torch.bool), diagonal=-1)
    A = A.masked_fill(~mask, 0)
    
    # fla convention: reshape to [B, H, T, C] (treating NT*C as T, last dim is chunk_size)
    # Actually looking at solve_tril, it expects [B, T, H, BT] with head_first=False
    # But within our function we work in head_first=True.
    # Let's return [B, H, NT*C, C] = [B, H, T_padded, C] then trim:
    A = A.reshape(B, H, T_padded, C)
    
    if pad > 0:
        A = A[:, :, :T, :]
    
    return A.to(output_dtype)


def _torch_solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    **kwargs,
) -> torch.Tensor:
    """Pure PyTorch solve_tril: compute (I + A)^{-1} where A is strictly lower triangular.
    
    Input A: [B, H, T, C] where T = NT * C (chunk-aligned)
    Output: same shape, (I + A)^{-1} per chunk.
    Results are cast to fp16 to match fla's triton kernel behavior (prevents numerical explosion).
    """
    B, H, T, C = A.shape
    NT = T // C
    
    # Reshape to [B*H*NT, C, C]
    A_blocks = A.reshape(B, H, NT, C, C)
    A_flat = A_blocks.reshape(-1, C, C)
    
    # I + A is unit lower triangular; compute its inverse
    I = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0).expand_as(A_flat)
    IpA = I + A_flat
    
    # Solve (I+A) * X = I  =>  X = (I+A)^{-1}
    Ai = torch.linalg.solve_triangular(IpA, I, upper=False, unitriangular=True)
    
    # Critical: fla's triton kernel stores result in fp16, which naturally clips
    # extreme values. Replace Inf/NaN with 0 then clamp to fp16 range.
    Ai = torch.nan_to_num(Ai, nan=0.0, posinf=65504.0, neginf=-65504.0)
    Ai = Ai.clamp(-65504.0, 65504.0)
    
    # Reshape back
    Ai = Ai.reshape(B, H, NT, C, C)
    Ai = Ai.reshape(B, H, T, C)
    
    return Ai


def _ensure_npu_ops() -> bool:
    """Lazy initialization of NPU FLA ops. Call once before first use."""
    global _NPU_OPS_AVAILABLE, _chunk_local_cumsum, _chunk_scaled_dot_kkt_fwd, _solve_tril, _l2norm

    if _NPU_OPS_AVAILABLE is not None:
        return _NPU_OPS_AVAILABLE

    try:
        import fla_npu  # noqa: F401 — loads .so and registers torch.ops.npu.*

        # Use triton-on-NPU ops from fla_npu (faster, matches fla numerical behavior)
        from fla_npu.ops.triton import (
            chunk_local_cumsum_scalar as _triton_cumsum,
            chunk_scaled_dot_kkt_fwd as _triton_kkt,
            solve_tril_npu as _triton_solve,
        )

        _chunk_local_cumsum = _triton_cumsum
        _chunk_scaled_dot_kkt_fwd = _triton_kkt
        _solve_tril = _triton_solve
        _l2norm = None  # l2norm handled separately in gated_delta_net.py

        # Verify AscendC ops are registered
        required_ops = [
            'npu_chunk_gated_delta_rule_fwd_h',
            'npu_chunk_gated_delta_rule_bwd_dhu',
            'npu_recompute_w_u_fwd',
            'npu_chunk_fwd_o',
            'npu_chunk_bwd_dv_local',
            'npu_chunk_bwd_dqkwg',
            'npu_prepare_wy_repr_bwd_da',
            'npu_prepare_wy_repr_bwd_full',
        ]
        for op_name in required_ops:
            assert hasattr(torch.ops.npu, op_name), f"{op_name} not registered"

        _NPU_OPS_AVAILABLE = True
        logger.info("NPU FLA AscendC ops loaded successfully for GDN.")
        return True

    except (ImportError, AssertionError) as e:
        logger.warning(f"NPU FLA ops not available: {e}")
        _NPU_OPS_AVAILABLE = False
        return False


def _to_cu_seqlens_list(cu_seqlens: Optional[torch.Tensor]) -> Optional[list]:
    """Convert cu_seqlens tensor to list[int] for NPU ops."""
    if cu_seqlens is None:
        return None
    return cu_seqlens.cpu().tolist()


def _compute_chunk_indices(
    cu_seqlens: Optional[torch.Tensor], T: int, chunk_size: int
) -> Optional[list]:
    """Compute flattened chunk_indices list for NPU ops.

    chunk_indices is a flat list: [seq_idx_0, chunk_idx_0, seq_idx_1, chunk_idx_1, ...]
    Each pair maps a global chunk to (sequence_index, local_chunk_index_within_sequence).
    """
    if cu_seqlens is None:
        return None

    cu_seqlens_list = cu_seqlens.cpu().tolist()
    indices = []
    for seq_idx in range(len(cu_seqlens_list) - 1):
        seq_start = cu_seqlens_list[seq_idx]
        seq_end = cu_seqlens_list[seq_idx + 1]
        seq_len = seq_end - seq_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            indices.append(seq_idx)
            indices.append(chunk_idx)
    return indices


class NPUChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """
    Autograd Function that implements chunk_gated_delta_rule using NPU AscendC kernels.

    Accepts the SAME tensor format as fla's chunk_gated_delta_rule:
        q, k, v: [B, H, T, K/V]  (already transposed by caller or internally)
        g, beta:  [B, H, T]
    
    NPU AscendC ops also expect [B, H, T, D] — no additional transpose needed.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.Tensor],
        chunk_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # q, k: [B, H, T, K], v: [B, H, T, V], g/beta: [B, H, T]
        B, H, T, K = q.shape
        V = v.shape[-1]

        cu_seqlens_list = _to_cu_seqlens_list(cu_seqlens)
        chunk_indices = _compute_chunk_indices(cu_seqlens, T, chunk_size)

        # Step 1: chunk_local_cumsum(g) — cumulative gate within each chunk
        # triton op expects [B, T, H], our g is [B, H, T]
        g_bth = g.transpose(1, 2).contiguous()  # [B, H, T] -> [B, T, H]
        g_cumsum_bth = _chunk_local_cumsum(g_bth, chunk_size=chunk_size)
        g_cumsum = g_cumsum_bth.transpose(1, 2).contiguous()  # -> [B, H, T]

        # Step 2: chunk_scaled_dot_kkt — compute A matrix
        # triton op: k=[B,H,T,K], g=[B,T,H], beta=[B,T,H] -> A=[B,T,H,C]
        beta_bth = beta.transpose(1, 2).contiguous()  # [B, H, T] -> [B, T, H]
        A_bthc = _chunk_scaled_dot_kkt_fwd(
            k, g=g_cumsum_bth, beta=beta_bth,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )

        # Step 3: solve_tril(A) — triangular solve
        # triton op: A=[B,T,H,C] -> Ai=[B,T,H,C]
        A_bthc = _solve_tril(A_bthc, cu_seqlens=cu_seqlens)
        # Transpose to head_first [B, H, T, C] for AscendC ops
        A = A_bthc.transpose(1, 2).contiguous()

        # Step 4: recompute_w_u_fwd (AscendC)
        # NOTE: g is REQUIRED by the AscendC kernel despite being declared optional
        A_compute = A.to(k.dtype)  # Cast A to match k's dtype (fp16/bf16)
        g_compute = g_cumsum.to(k.dtype)
        w, u = torch.ops.npu.npu_recompute_w_u_fwd(
            k, v, beta, A_compute, chunk_size,
            g=g_compute, gk=None, cu_seqlens=cu_seqlens_list, chunk_indices=chunk_indices,
        )

        # Step 5: chunk_gated_delta_rule_fwd_h (AscendC)
        # Signature: (k, w, u, g=None, *, initial_state=None, chunk_size=None, ...)
        # Returns: (h, v_new, final_state)
        h, v_new, final_state_out = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g_cumsum,
            initial_state=initial_state,
            chunk_size=chunk_size,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )

        # Use final_state from op output
        final_state = final_state_out if output_final_state else None

        # Step 6: chunk_fwd_o (AscendC)
        o = torch.ops.npu.npu_chunk_fwd_o(
            q, k, v_new, h, scale,
            g=g_cumsum,
            g_gamma=None,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            transpose_state_layout=False,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, g_cumsum, beta, A, initial_state)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.output_final_state = output_final_state

        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, d_final_state):
        q, k, v, g, beta, A, initial_state = ctx.saved_tensors
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        cu_seqlens_list = ctx.cu_seqlens_list
        chunk_indices = ctx.chunk_indices

        # Recompute w, u (activation recomputation to save memory)
        A_compute = A.to(k.dtype)
        g_compute = g.to(k.dtype)
        w, u = torch.ops.npu.npu_recompute_w_u_fwd(
            k, v, beta, A_compute, chunk_size,
            g=g_compute, gk=None, cu_seqlens=cu_seqlens_list, chunk_indices=chunk_indices,
        )

        # Recompute h, v_new
        h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k, w, u, g,
            initial_state=initial_state,
            chunk_size=chunk_size,
            output_final_state=False,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )

        # Step 1: chunk_bwd_dv_local (AscendC)
        dv = torch.ops.npu.npu_chunk_bwd_dv_local(
            q, k, do, g,
            scale=scale,
            chunk_size=chunk_size,
            g_gamma=None,
            A=None,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )

        # Step 2: chunk_gated_delta_rule_bwd_dhu (AscendC)
        dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            q, k, w, do, dv,
            scale=scale,
            chunk_size=chunk_size,
            g=g,
            gK=None,
            h0=initial_state,
            dht=d_final_state,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )
        # dv2 is the updated dv with cross-chunk contributions
        dv = dv2

        # Step 3: chunk_bwd_dqkwg (AscendC)
        dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
            q, k, v_new, g, h, do, dh, dv, chunk_size,
            cu_seqlens=cu_seqlens_list,
            w=w,
            g_gamma=None,
            chunk_indices=chunk_indices,
            scale=scale,
            use_exp2=None,
            transpose_state_layout=None,
        )

        # Step 4a: compute dA (AscendC)
        # NOTE: PrepareWyReprBwdDa does NOT support bf16 — cast to fp16 locally
        _cast = k.dtype == torch.bfloat16
        if _cast:
            _k, _v, _beta, _A, _dw, _du, _g = (
                k.half(), v.half(), beta.half(), A.half(), dw.half(), dv.half(), g.half()
            )
        else:
            _k, _v, _beta, _A, _dw, _du, _g = k, v, beta, A, dw, dv, g
        dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
            _k, _v, _beta, _A, _dw, _du, _g,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )
        if _cast:
            dA = dA.to(torch.bfloat16)

        # Step 4b: compute dk2, dv_final, dbeta, dg2 (AscendC)
        # NOTE: PrepareWyReprBwdFull may also not support bf16
        if _cast:
            _dA = dA.half()
            dk2, dv_final, dbeta, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
                _k, _v, _beta, _A, _dA, _dw, _du, _g, chunk_size,
                cu_seqlens=cu_seqlens_list,
                chunk_indices=chunk_indices,
            )
            dk2, dv_final, dbeta, dg2 = dk2.bfloat16(), dv_final.bfloat16(), dbeta.bfloat16(), dg2.bfloat16()
        else:
            dk2, dv_final, dbeta, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
                k, v, beta, A, dA, dw, dv, g, chunk_size,
                cu_seqlens=cu_seqlens_list,
                chunk_indices=chunk_indices,
            )

        # Accumulate gradients
        dk = dk + dk2
        dg = dg + dg2

        # Step 5: reverse cumsum for dg
        # triton cumsum expects [B, T, H], dg is [B, H, T]
        dg_bth = dg.transpose(1, 2).contiguous()
        dg_bth = _chunk_local_cumsum(
            dg_bth, chunk_size=chunk_size, reverse=True,
            cu_seqlens=cu_seqlens,
        )
        dg = dg_bth.transpose(1, 2).contiguous()

        # Return gradients matching forward args:
        # q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, chunk_size
        return dq, dk, dv_final, dg, dbeta, None, dh0, None, None, None


def npu_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    NPU-accelerated chunk gated delta rule.

    Drop-in replacement for fla's chunk_gated_delta_rule on Ascend NPU.
    Uses AscendC custom kernels for compute-heavy ops (fwd_h, bwd_dhu, fwd_o, etc.)
    and triton-on-NPU for lighter ops (cumsum, kkt, solve_tril).

    Matches fla's public API signature so it can be used as a direct replacement
    in GatedDeltaNet.forward().

    Args:
        q: [B, T, H, K] or [B, H, T, K] if head_first=True
        k: [B, T, H, K] or [B, H, T, K]
        v: [B, T, H, V] or [B, H, T, V]
        g: [B, T, H] or [B, H, T]  — gates (log-space, pre-cumsum)
        beta: [B, T, H] or [B, H, T]
        scale: attention scale (default: 1/sqrt(K))
        initial_state: [B, H, K, V] or None
        output_final_state: whether to return final recurrent state
        cu_seqlens: [num_seqs+1] cumulative seq lengths for varlen
        chunk_size: chunk size (default 64)
        head_first: if True, inputs are [B, H, T, D]; if False, [B, T, H, D]
        use_qk_l2norm_in_kernel: if True, apply l2norm to q/k inside
                                 (ignored here — already done in Megatron)

    Returns:
        o: same layout as q (either [B, T, H, V] or [B, H, T, V])
        final_state: [B, H, K, V] or None
    """
    if not _ensure_npu_ops():
        raise RuntimeError(
            "NPU FLA ops not available. Install fla_npu package and ensure "
            "AscendC kernels are built."
        )

    # Transpose to head_first [B, H, T, D] for NPU ops
    if not head_first:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()

    if scale is None:
        scale = q.shape[-1] ** -0.5

    o, final_state = NPUChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        cu_seqlens, chunk_size,
    )

    # Transpose output back to [B, T, H, V] if not head_first
    if not head_first:
        o = o.transpose(1, 2).contiguous()

    return o, final_state
