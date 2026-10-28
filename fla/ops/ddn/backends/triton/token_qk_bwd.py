# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def ddn_token_qk_bwd_kernel(
    q_normalized, k_normalized, beta, q_rstd, k_rstd,
    dq_normalized, dk_normalized, dbeta_k,
    dq, dk, dbeta,
    cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr,
    PACKED: tl.constexpr, NORMALIZE_QK: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    row = tl.arange(0, 32)
    key = tl.arange(0, BK)
    token = bos + row
    source = (token[:, None] * H + head) * DK + key[None, :]
    source_mask = (token[:, None] < eos) & (key[None, :] < DK)
    packed = (chunk * 32 + row[:, None]) * DK + key[None, :]
    packed_mask = key[None, :] < DK
    q_normalized_value = tl.load(q_normalized + packed, packed_mask, 0).to(tl.float32)
    k_normalized_value = tl.load(k_normalized + packed, packed_mask, 0).to(tl.float32)
    dq_value = tl.load(dq_normalized + packed, packed_mask, 0).to(tl.float32)
    dk_value = tl.load(dk_normalized + packed, packed_mask, 0).to(tl.float32)
    dbeta_k_value = tl.load(dbeta_k + packed, packed_mask, 0).to(tl.float32)
    beta_value = tl.load(beta + token * H + head, token < eos, 0).to(tl.float32)
    dbeta_value = tl.sum(dbeta_k_value * k_normalized_value, 1)
    dk_value += beta_value[:, None] * dbeta_k_value
    if NORMALIZE_QK:
        q_rstd_value = tl.load(q_rstd + chunk * 32 + row, token < eos, 0).to(tl.float32)
        k_rstd_value = tl.load(k_rstd + chunk * 32 + row, token < eos, 0).to(tl.float32)
        if DK == 1:
            # BF16 rounds scalar normalized values to +/-1; use epsilon to avoid cancellation.
            dq_value *= (1e-6 * q_rstd_value * q_rstd_value * q_rstd_value)[:, None]
            dk_value *= (1e-6 * k_rstd_value * k_rstd_value * k_rstd_value)[:, None]
        else:
            dq_value = q_rstd_value[:, None] * (
                dq_value - q_normalized_value * tl.sum(q_normalized_value * dq_value, 1)[:, None]
            )
            dk_value = k_rstd_value[:, None] * (
                dk_value - k_normalized_value * tl.sum(k_normalized_value * dk_value, 1)[:, None]
            )
    tl.store(dq + source, dq_value, source_mask)
    tl.store(dk + source, dk_value, source_mask)
    tl.store(dbeta + token * H + head, dbeta_value, token < eos)


def token_qk_bwd(
    q_normalized, k_normalized, beta, q_rstd, k_rstd,
    dq_normalized, dk_normalized, dbeta_k, layout, normalize_qk,
):
    dk_dim = q_normalized.shape[-1]
    dq = torch.empty(layout.batch, layout.length, layout.heads, dk_dim, device=q_normalized.device, dtype=torch.bfloat16)
    dk = torch.empty_like(dq)
    dbeta = torch.empty_like(beta)
    ddn_token_qk_bwd_kernel[(layout.chunks * layout.heads,)](
        q_normalized, k_normalized, beta, q_rstd, k_rstd,
        dq_normalized, dk_normalized, dbeta_k,
        dq, dk, dbeta,
        layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dk_dim, max(16, triton.next_power_of_2(dk_dim)),
        layout.cu_seqlens is not None, normalize_qk,
        num_warps=4,
    )
    return dq, dk, dbeta
