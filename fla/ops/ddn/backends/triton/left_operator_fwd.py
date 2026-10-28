# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.backends.triton.left_inverse_fwd import left_inverse_fwd
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=1) for warps in (2, 4)],
    key=['DK'],
    **autotune_cache_kwargs,
)
@triton.jit
def ddn_left_operator_fwd_kernel(
    Q_pi, bar_U, bar_K_Pi_C, M_L, tril_Q_pi_bar_K_T,
    G_L, Pi_C_H, Q_eff,
    DK: tl.constexpr, BK: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, 32)
    matrix_offset = chunk * 1024 + row[:, None] * 32 + row[None, :]
    m = tl.load(M_L + matrix_offset).to(tl.bfloat16)
    interaction = tl.load(tril_Q_pi_bar_K_T + matrix_offset).to(tl.bfloat16)
    g = tl.dot(interaction, m).to(tl.bfloat16)
    tl.store(G_L + matrix_offset, g)

    for block in range(tl.cdiv(DK, BK)):
        key = block * BK + tl.arange(0, BK)
        offset = (chunk * 32 + row[:, None]) * DK + key[None, :]
        mask = key[None, :] < DK
        k = tl.load(bar_K_Pi_C + offset, mask, 0).to(tl.bfloat16)
        p = tl.dot(tl.trans(k), m)
        tl.store(Pi_C_H + (chunk * DK + key[:, None]) * 32 + row[None, :], p, key[:, None] < DK)
        u = tl.load(bar_U + offset, mask, 0).to(tl.bfloat16)
        q = tl.load(Q_pi + offset, mask, 0).to(tl.float32)
        q_eff = q - tl.dot(g, u)
        tl.store(Q_eff + offset, q_eff, mask)


def left_operator_fwd(
    q_normalized,
    k_normalized,
    beta,
    log_alpha,
    Q_pi,
    bar_U,
    bar_K_Pi_C,
    layout,
    scale,
    save_for_backward,
    M_L=None,
    tril_Q_pi_bar_K_T=None,
):
    dk = q_normalized.shape[-1]
    matrix_dtype = bar_U.dtype
    compute_inverse = M_L is None
    if compute_inverse:
        M_L, tril_Q_pi_bar_K_T = left_inverse_fwd(
            q_normalized, k_normalized, beta, log_alpha, layout, scale,
        )
    elif tril_Q_pi_bar_K_T is None:
        raise ValueError("Cached M_L requires the cached left interaction matrix.")
    G_L = torch.empty(M_L.shape, device=M_L.device, dtype=matrix_dtype)
    Pi_C_H = torch.empty(
        layout.chunks, layout.heads, dk, 32,
        device=q_normalized.device, dtype=matrix_dtype,
    )
    Q_eff = torch.empty_like(Q_pi)
    grid = (layout.chunks * layout.heads,)
    ddn_left_operator_fwd_kernel[grid](
        Q_pi, bar_U, bar_K_Pi_C, M_L, tril_Q_pi_bar_K_T, G_L, Pi_C_H, Q_eff,
        dk, 64,
    )
    if compute_inverse and not save_for_backward:
        tril_Q_pi_bar_K_T = None
    return M_L, G_L, Pi_C_H, Q_eff, tril_Q_pi_bar_K_T
