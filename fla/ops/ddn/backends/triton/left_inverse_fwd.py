# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.backends.triton.solve_tril import solve_tril
from fla.ops.ddn.layout import chunk_bounds
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs


@triton.jit
def _build_left_interactions(
    q_normalized, k_normalized, beta, log_alpha,
    chunk, head, bos, eos,
    H: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.arange(0, 16)
    d = tl.arange(0, BK)
    token0, token1 = bos + row, bos + 16 + row
    key_mask = d[None, :] < DK
    source0 = (token0[:, None] * H + head) * DK + d[None, :]
    source1 = (token1[:, None] * H + head) * DK + d[None, :]
    log_alpha0 = tl.load(
        log_alpha + source0, (token0[:, None] < eos) & key_mask, 0
    ).to(tl.float32)
    log_alpha1 = tl.load(
        log_alpha + source1, (token1[:, None] < eos) & key_mask, 0
    ).to(tl.float32)
    log_pi_i0 = tl.cumsum(log_alpha0, 0)
    log_pi_i1 = tl.sum(log_alpha0, 0)[None, :] + tl.cumsum(log_alpha1, 0)
    log_pi_max = tl.maximum(tl.max(log_pi_i0, 0), tl.max(log_pi_i1, 0))
    log_pi_min = tl.minimum(tl.min(log_pi_i0, 0), tl.min(log_pi_i1, 0))

    Q_pi_bar_K_T_00 = tl.zeros((16, 16), tl.float32)
    Q_pi_bar_K_T_10 = tl.zeros((16, 16), tl.float32)
    Q_pi_bar_K_T_11 = tl.zeros((16, 16), tl.float32)
    T_L_00 = tl.zeros((16, 16), tl.float32)
    T_L_10 = tl.zeros((16, 16), tl.float32)
    T_L_11 = tl.zeros((16, 16), tl.float32)
    if tl.max(log_pi_max - log_pi_min, 0) <= 60:
        log_pi_center = (log_pi_max + log_pi_min) * 0.5
        pi_i_div_pi_center0 = tl.exp(log_pi_i0 - log_pi_center[None, :])
        pi_i_div_pi_center1 = tl.exp(log_pi_i1 - log_pi_center[None, :])
        pi_center_div_pi_i0 = tl.exp(log_pi_center[None, :] - log_pi_i0)
        pi_center_div_pi_i1 = tl.exp(log_pi_center[None, :] - log_pi_i1)
        key_offset0 = (chunk * 32 + row[:, None]) * DK + d[None, :]
        key_offset1 = (chunk * 32 + 16 + row[:, None]) * DK + d[None, :]
        k0 = tl.load(k_normalized + key_offset0, key_mask, 0).to(tl.float32)
        k1 = tl.load(k_normalized + key_offset1, key_mask, 0).to(tl.float32)
        beta0 = tl.load(beta + token0 * H + head, token0 < eos, 0).to(tl.float32)
        beta1 = tl.load(beta + token1 * H + head, token1 < eos, 0).to(tl.float32)
        beta_k0 = (beta0[:, None] * k0).to(k_normalized.dtype.element_ty).to(tl.float32)
        beta_k1 = (beta1[:, None] * k1).to(k_normalized.dtype.element_ty).to(tl.float32)
        pi_center_div_pi_i_k0 = k0 * pi_center_div_pi_i0
        pi_center_div_pi_i_k1 = k1 * pi_center_div_pi_i1
        pi_i_div_pi_center_q0 = (
            tl.load(q_normalized + key_offset0, key_mask, 0).to(tl.float32)
            * pi_i_div_pi_center0
        )
        pi_i_div_pi_center_q1 = (
            tl.load(q_normalized + key_offset1, key_mask, 0).to(tl.float32)
            * pi_i_div_pi_center1
        )
        beta_pi_i_div_pi_center_k0 = beta_k0 * pi_i_div_pi_center0
        beta_pi_i_div_pi_center_k1 = beta_k1 * pi_i_div_pi_center1
        Q_pi_bar_K_T_00 = tl.dot(
            pi_i_div_pi_center_q0.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k0.to(tl.bfloat16)),
        ) * SCALE
        Q_pi_bar_K_T_10 = tl.dot(
            pi_i_div_pi_center_q1.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k0.to(tl.bfloat16)),
        ) * SCALE
        Q_pi_bar_K_T_11 = tl.dot(
            pi_i_div_pi_center_q1.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k1.to(tl.bfloat16)),
        ) * SCALE
        T_L_00 = tl.dot(
            beta_pi_i_div_pi_center_k0.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k0.to(tl.bfloat16)),
        )
        T_L_10 = tl.dot(
            beta_pi_i_div_pi_center_k1.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k0.to(tl.bfloat16)),
        )
        T_L_11 = tl.dot(
            beta_pi_i_div_pi_center_k1.to(tl.bfloat16),
            tl.trans(pi_center_div_pi_i_k1.to(tl.bfloat16)),
        )
        Q_pi_bar_K_T_00 = tl.where(row[:, None] >= row[None, :], Q_pi_bar_K_T_00, 0)
        Q_pi_bar_K_T_11 = tl.where(row[:, None] >= row[None, :], Q_pi_bar_K_T_11, 0)
        T_L_00 = tl.where(row[:, None] > row[None, :], T_L_00, 0)
        T_L_11 = tl.where(row[:, None] > row[None, :], T_L_11, 0)
    else:
        key_offset0 = (chunk * 32 + row[:, None]) * DK + d[None, :]
        key_offset1 = (chunk * 32 + 16 + row[:, None]) * DK + d[None, :]
        k_normalized0 = tl.load(k_normalized + key_offset0, key_mask, 0).to(tl.float32)
        k_normalized1 = tl.load(k_normalized + key_offset1, key_mask, 0).to(tl.float32)
        for current in range(16):
            q_normalized0 = tl.load(
                q_normalized + (chunk * 32 + current) * DK + d, d < DK, 0
            ).to(tl.float32)
            beta0 = tl.load(beta + (bos + current) * H + head, bos + current < eos, 0).to(tl.float32)
            k0 = tl.load(k_normalized + (chunk * 32 + current) * DK + d, d < DK, 0).to(tl.float32)
            beta_k0 = (beta0 * k0).to(k_normalized.dtype.element_ty).to(tl.float32)
            log_pi_i_current0 = tl.sum(
                tl.where(row[:, None] == current, log_pi_i0, 0), 0
            )
            pi_i_pi_j_inv_k00 = k_normalized0 * tl.exp(
                tl.where(
                    row[:, None] <= current,
                    log_pi_i_current0[None, :] - log_pi_i0,
                    0,
                )
            )
            Q_pi_bar_K_T_row00 = (
                tl.sum(pi_i_pi_j_inv_k00 * q_normalized0[None, :], 1) * SCALE
            )
            T_L_row00 = tl.sum(pi_i_pi_j_inv_k00 * beta_k0[None, :], 1)
            Q_pi_bar_K_T_00 = tl.where(
                (row == current)[:, None],
                tl.where(row[None, :] <= current, Q_pi_bar_K_T_row00[None, :], 0),
                Q_pi_bar_K_T_00,
            )
            T_L_00 = tl.where(
                (row == current)[:, None],
                tl.where(row[None, :] < current, T_L_row00[None, :], 0),
                T_L_00,
            )

            q_normalized1 = tl.load(
                q_normalized + (chunk * 32 + 16 + current) * DK + d,
                d < DK,
                0,
            ).to(tl.float32)
            beta1 = tl.load(beta + (bos + 16 + current) * H + head, bos + 16 + current < eos, 0).to(tl.float32)
            k1 = tl.load(k_normalized + (chunk * 32 + 16 + current) * DK + d, d < DK, 0).to(tl.float32)
            beta_k1 = (beta1 * k1).to(k_normalized.dtype.element_ty).to(tl.float32)
            log_pi_i_current1 = tl.sum(
                tl.where(row[:, None] == current, log_pi_i1, 0), 0
            )
            pi_i_pi_j_inv_k10 = k_normalized0 * tl.exp(
                log_pi_i_current1[None, :] - log_pi_i0
            )
            pi_i_pi_j_inv_k11 = k_normalized1 * tl.exp(
                tl.where(
                    row[:, None] <= current,
                    log_pi_i_current1[None, :] - log_pi_i1,
                    0,
                )
            )
            Q_pi_bar_K_T_row10 = (
                tl.sum(pi_i_pi_j_inv_k10 * q_normalized1[None, :], 1) * SCALE
            )
            T_L_row10 = tl.sum(pi_i_pi_j_inv_k10 * beta_k1[None, :], 1)
            Q_pi_bar_K_T_row11 = (
                tl.sum(pi_i_pi_j_inv_k11 * q_normalized1[None, :], 1) * SCALE
            )
            T_L_row11 = tl.sum(pi_i_pi_j_inv_k11 * beta_k1[None, :], 1)
            Q_pi_bar_K_T_10 = tl.where(
                (row == current)[:, None], Q_pi_bar_K_T_row10[None, :], Q_pi_bar_K_T_10
            )
            T_L_10 = tl.where((row == current)[:, None], T_L_row10[None, :], T_L_10)
            Q_pi_bar_K_T_11 = tl.where(
                (row == current)[:, None],
                tl.where(row[None, :] <= current, Q_pi_bar_K_T_row11[None, :], 0),
                Q_pi_bar_K_T_11,
            )
            T_L_11 = tl.where(
                (row == current)[:, None],
                tl.where(row[None, :] < current, T_L_row11[None, :], 0),
                T_L_11,
            )

    return Q_pi_bar_K_T_00, Q_pi_bar_K_T_10, Q_pi_bar_K_T_11, T_L_00, T_L_10, T_L_11


@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=1) for warps in (2, 4)],
    key=['H', 'T', 'DK', 'PACKED'],
    **autotune_cache_kwargs,
)
@triton.jit
def ddn_left_inverse_fwd_kernel(
    q_normalized, k_normalized, beta, log_alpha,
    M_L, tril_Q_pi_bar_K_T, cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr,
    PACKED: tl.constexpr, SCALE: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    (
        Q_pi_bar_K_T_00, Q_pi_bar_K_T_10, Q_pi_bar_K_T_11,
        T_L_00, T_L_10, T_L_11,
    ) = _build_left_interactions(
        q_normalized, k_normalized, beta, log_alpha,
        chunk, head, bos, eos, H, DK, BK, SCALE,
    )
    M_L_00, M_L_10, M_L_11 = solve_tril(T_L_00, T_L_10, T_L_11, 'tf32')
    row = tl.arange(0, 16)
    offset = chunk * 1024 + row[:, None] * 32 + row[None, :]
    zeros = tl.zeros((16, 16), tl.float32)
    tl.store(M_L + offset, M_L_00)
    tl.store(M_L + offset + 16, zeros)
    tl.store(M_L + offset + 16 * 32, M_L_10)
    tl.store(M_L + offset + 16 * 33, M_L_11)
    tl.store(tril_Q_pi_bar_K_T + offset, Q_pi_bar_K_T_00)
    tl.store(tril_Q_pi_bar_K_T + offset + 16, zeros)
    tl.store(tril_Q_pi_bar_K_T + offset + 16 * 32, Q_pi_bar_K_T_10)
    tl.store(tril_Q_pi_bar_K_T + offset + 16 * 33, Q_pi_bar_K_T_11)


def left_inverse_fwd(q_normalized, k_normalized, beta, log_alpha, layout, scale):
    dk = q_normalized.shape[-1]
    M_L = torch.empty(
        layout.chunks, layout.heads, 32, 32,
        device=q_normalized.device, dtype=torch.bfloat16,
    )
    tril_Q_pi_bar_K_T = torch.empty_like(M_L)
    ddn_left_inverse_fwd_kernel[(layout.chunks * layout.heads,)](
        q_normalized, k_normalized, beta, log_alpha,
        M_L, tril_Q_pi_bar_K_T, layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dk, max(16, triton.next_power_of_2(dk)),
        layout.cu_seqlens is not None, scale,
    )
    return M_L, tril_Q_pi_bar_K_T
