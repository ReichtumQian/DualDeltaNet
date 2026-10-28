# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def _sum_values(x, LOG_BD: tl.constexpr):
    """Use the same reduction order with raw and cached normalized inputs."""
    for step in tl.static_range(LOG_BD):
        x = tl.sum(tl.reshape(x, (32, 1 << (LOG_BD - step - 1), 2)), 2)
    return tl.reshape(x, (32,))


@triton.jit
def ddn_token_preparation_fwd_kernel(
    q, k, v, log_alpha, beta_logits, gamma_logits, m_logits,
    q_normalized, k_normalized,
    Q_pi, bar_U, bar_K_Pi_C,
    beta, gamma, eta,
    log_pi_C, E, hat_V, q_rstd, k_rstd, v_rstd,
    cu_seqlens, indices,
    BETA_B: tl.constexpr, BETA_T: tl.constexpr, BETA_H: tl.constexpr,
    GAMMA_B: tl.constexpr, GAMMA_T: tl.constexpr, GAMMA_H: tl.constexpr,
    H: tl.constexpr, T: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, BD: tl.constexpr, LOG_BD: tl.constexpr,
    PACKED: tl.constexpr, NORM: tl.constexpr, RECOMPUTE: tl.constexpr,
    EPS: tl.constexpr, SCALE: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    row = tl.arange(0, 32)
    d = tl.arange(0, BD)
    token = bos + row
    key_source = (token[:, None] * H + head) * DK + d[None, :]
    key_source_mask = (token[:, None] < eos) & (d[None, :] < DK)
    key_offset = (chunk * 32 + row[:, None]) * DK + d[None, :]
    key_mask = d[None, :] < DK
    if RECOMPUTE:
        beta_value = tl.load(beta + token * H + head, token < eos, 0).to(tl.float32)
        gamma_value = tl.load(gamma + chunk * 32 + row).to(tl.float32)
        q_normalized_value = tl.load(q_normalized + key_offset, key_mask, 0).to(tl.float32)
        k_normalized_value = tl.load(k_normalized + key_offset, key_mask, 0).to(tl.float32)
    else:
        beta_offset = (token // T) * BETA_B + (token % T) * BETA_T + head * BETA_H
        gamma_offset = (token // T) * GAMMA_B + (token % T) * GAMMA_T + head * GAMMA_H
        beta_value = tl.sigmoid(tl.load(beta_logits + beta_offset, token < eos, 0).to(tl.float32))
        tl.store(beta + token * H + head, beta_value, token < eos)
        gamma_value = tl.sigmoid(
            tl.load(gamma_logits + gamma_offset, token < eos, 0).to(tl.float32)
        )

        q_normalized_value = tl.load(q + key_source, key_source_mask, 0).to(tl.float32)
        k_normalized_value = tl.load(k + key_source, key_source_mask, 0).to(tl.float32)
        if NORM:
            q_rstd_value = tl.rsqrt(
                tl.sum(q_normalized_value * q_normalized_value, 1) + 1e-6
            )
            k_rstd_value = tl.rsqrt(
                tl.sum(k_normalized_value * k_normalized_value, 1) + 1e-6
            )
            q_normalized_value *= q_rstd_value[:, None]
            k_normalized_value *= k_rstd_value[:, None]
            tl.store(q_rstd + chunk * 32 + row, q_rstd_value)
            tl.store(k_rstd + chunk * 32 + row, k_rstd_value)

        tl.store(q_normalized + key_offset, q_normalized_value, key_mask)
        tl.store(k_normalized + key_offset, k_normalized_value, key_mask)

        q_normalized_value = q_normalized_value.to(q_normalized.dtype.element_ty).to(tl.float32)
        k_normalized_value = k_normalized_value.to(k_normalized.dtype.element_ty).to(tl.float32)
    beta_k_value = beta_value[:, None] * k_normalized_value

    log_alpha_value = tl.load(log_alpha + key_source, key_source_mask, 0).to(tl.float32)
    log_pi_i = tl.cumsum(log_alpha_value, 0)
    log_pi_C_value = tl.sum(log_alpha_value, 0)
    pi_i = tl.exp(log_pi_i)
    tl.store(log_pi_C + chunk * DK + d, log_pi_C_value, d < DK)
    tl.store(Q_pi + key_offset, q_normalized_value * pi_i * SCALE, key_mask)
    tl.store(bar_U + key_offset, beta_k_value * pi_i, key_mask)
    tl.store(
        bar_K_Pi_C + key_offset,
        k_normalized_value * tl.exp(log_pi_C_value[None, :] - log_pi_i),
        key_mask,
    )
    if not RECOMPUTE:
        tl.store(gamma + chunk * 32 + row, tl.where(token < eos, gamma_value, 0))

    value_source = (token[:, None] * H + head) * DV + d[None, :]
    value_source_mask = (token[:, None] < eos) & (d[None, :] < DV)
    value_offset = (chunk * 32 + row[:, None]) * DV + d[None, :]
    value_mask = d[None, :] < DV
    if RECOMPUTE:
        hat_V_value = tl.load(hat_V + value_offset, value_mask, 0).to(tl.float32)
        v_rstd_value = tl.load(v_rstd + chunk * 32 + row).to(tl.float32)
    else:
        v_value = tl.load(v + value_source, value_source_mask, 0).to(tl.float32)
        nu = tl.sqrt(tl.sum(v_value * v_value, 1) + EPS)
        hat_V_value = (v_value / nu[:, None]).to(hat_V.dtype.element_ty).to(tl.float32)
        v_rstd_value = 1.0 / nu
        tl.store(hat_V + value_offset, hat_V_value, value_mask)
        tl.store(v_rstd + chunk * 32 + row, v_rstd_value)
    m = tl.sigmoid(tl.load(m_logits + value_source, value_source_mask, 0).to(tl.float32))
    m_hat_V = m * hat_V_value
    E_value = m_hat_V * tl.rsqrt(_sum_values(m_hat_V * m_hat_V, LOG_BD) + EPS)[:, None]
    eta_value = (beta_value + gamma_value * (1 - beta_value) * _sum_values(hat_V_value * E_value, LOG_BD)) / v_rstd_value
    tl.store(E + value_offset, E_value, value_mask)
    tl.store(eta + chunk * 32 + row, tl.where(token < eos, eta_value, 0))


def token_preparation_fwd(
    q, k, v, log_alpha, beta_logits, gamma_logits, m_logits,
    layout, normalize_qk, eps, scale, cached_normalization=None,
):
    dk, dv = q.shape[-1], v.shape[-1]
    matrix_dtype = torch.float32 if min(dk, dv) < 16 else q.dtype
    recompute = cached_normalization is not None
    if recompute:
        q_normalized, k_normalized, hat_V, beta, gamma, q_rstd, k_rstd, v_rstd = cached_normalization
    else:
        q_normalized = torch.empty(*layout.shape, dk, device=q.device, dtype=torch.bfloat16)
        k_normalized = torch.empty_like(q_normalized)
        hat_V = torch.empty(*layout.shape, dv, device=q.device, dtype=torch.bfloat16)
        beta = torch.empty(beta_logits.shape, device=q.device, dtype=torch.float32)
        gamma = torch.empty(*layout.shape, device=q.device, dtype=torch.float32)
        q_rstd = torch.empty_like(gamma) if normalize_qk else None
        k_rstd = torch.empty_like(gamma) if normalize_qk else None
        v_rstd = torch.empty_like(gamma)
    Q_pi = torch.empty(*layout.shape, dk, device=q.device, dtype=matrix_dtype)
    bar_U = torch.empty_like(Q_pi)
    bar_K_Pi_C = torch.empty_like(Q_pi)
    log_pi_C = torch.empty(layout.chunks, layout.heads, dk, device=q.device, dtype=torch.float32)
    E = torch.empty(*layout.shape, dv, device=q.device, dtype=matrix_dtype)
    eta = torch.empty_like(gamma)
    bd = max(16, triton.next_power_of_2(max(dk, dv)))
    ddn_token_preparation_fwd_kernel[(layout.chunks * layout.heads,)](
        q, k, v, log_alpha, beta_logits, gamma_logits, m_logits,
        q_normalized, k_normalized, Q_pi, bar_U, bar_K_Pi_C,
        beta, gamma, eta, log_pi_C, E, hat_V, q_rstd, k_rstd, v_rstd,
        layout.cu_seqlens, layout.indices,
        *(beta_logits.stride()[:3] if beta_logits is not None else (0, 0, 0)),
        *(gamma_logits.stride()[:3] if gamma_logits is not None else (0, 0, 0)),
        layout.heads, layout.length, dk, dv, bd, bd.bit_length() - 1,
        layout.cu_seqlens is not None, normalize_qk, recompute, eps, scale,
        num_warps=4, enable_fp_fusion=False,
    )
    return (
        q_normalized, k_normalized, Q_pi, bar_U, bar_K_Pi_C, beta, gamma, eta,
        log_pi_C, E, hat_V, q_rstd, k_rstd, v_rstd,
    )
