# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def ddn_left_preparation_bwd_kernel(
    q_normalized, k_normalized, beta, log_alpha, log_pi_C,
    dT_L, dtril_Q_pi_bar_K_T, dQ_pi, dbar_U, dbar_K_Pi_C, dlog_pi_C,
    dq_normalized, dk_normalized, dbeta_k, dlog_alpha,
    cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr,
    PACKED: tl.constexpr, SCALE: tl.constexpr, MATRIX_BF16: tl.constexpr,
):
    matrix_dtype = tl.bfloat16 if MATRIX_BF16 else tl.float32
    program = tl.program_id(0).to(tl.int64)
    chunk = program // tl.cdiv(DK, BK)
    key = (program % tl.cdiv(DK, BK)) * BK + tl.arange(0, BK)
    row = tl.arange(0, 32)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    token = bos + row
    source = (token[:, None] * H + head) * DK + key[None, :]
    source_mask = (token[:, None] < eos) & (key[None, :] < DK)
    packed = (chunk * 32 + row[:, None]) * DK + key[None, :]
    packed_mask = key[None, :] < DK
    log_alpha_value = tl.load(log_alpha + source, source_mask, 0).to(tl.float32)
    log_pi_i = tl.cumsum(log_alpha_value, 0)
    beta_value = tl.load(beta + token * H + head, token < eos, 0).to(tl.float32)
    q_value = tl.load(q_normalized + packed, packed_mask, 0).to(matrix_dtype).to(tl.float32)
    k_value = tl.load(k_normalized + packed, packed_mask, 0).to(matrix_dtype).to(tl.float32)
    beta_k_value = (beta_value[:, None] * k_value).to(matrix_dtype).to(tl.float32)
    matrix = chunk * 1024 + row[:, None] * 32 + row[None, :]
    dtril_value = tl.where(
        row[:, None] >= row[None, :],
        tl.load(dtril_Q_pi_bar_K_T + matrix).to(tl.float32) * SCALE,
        0.0,
    )
    dT_L_value = tl.where(
        row[:, None] > row[None, :],
        tl.load(dT_L + matrix).to(tl.float32),
        0.0,
    )

    log_pi_high = tl.max(log_pi_i, 0)
    log_pi_low = tl.min(log_pi_i, 0)
    if tl.max(log_pi_high - log_pi_low, 0) <= 60:
        log_pi_center = (log_pi_high + log_pi_low) * 0.5
        pi_i_div_center = tl.exp(log_pi_i - log_pi_center[None, :])
        center_div_pi_i = tl.exp(log_pi_center[None, :] - log_pi_i)
        bar_K_center = k_value * center_div_pi_i
        dq_value = tl.dot(
            dtril_value.to(matrix_dtype),
            bar_K_center.to(matrix_dtype),
            input_precision='ieee',
        ) * pi_i_div_center
        dbeta_k_value = tl.dot(
            dT_L_value.to(matrix_dtype),
            bar_K_center.to(matrix_dtype),
            input_precision='ieee',
        ) * pi_i_div_center
        Q_pi_center = q_value * pi_i_div_center
        bar_U_center = beta_k_value * pi_i_div_center
        dk_value = (
            tl.dot(
                tl.trans(dtril_value).to(matrix_dtype),
                Q_pi_center.to(matrix_dtype),
                input_precision='ieee',
            )
            + tl.dot(
                tl.trans(dT_L_value).to(matrix_dtype),
                bar_U_center.to(matrix_dtype),
                input_precision='ieee',
            )
        ) * center_div_pi_i
    else:
        dq_value = tl.zeros((32, BK), tl.float32)
        dk_value = tl.zeros((32, BK), tl.float32)
        dbeta_k_value = tl.zeros((32, BK), tl.float32)
        for current in range(32):
            q_current = tl.load(
                q_normalized + (chunk * 32 + current) * DK + key, key < DK, 0,
            ).to(matrix_dtype).to(tl.float32)
            k_current = tl.load(
                k_normalized + (chunk * 32 + current) * DK + key, key < DK, 0,
            ).to(matrix_dtype).to(tl.float32)
            beta_current = tl.sum(tl.where(row == current, beta_value, 0.0), 0)
            beta_k_current = (beta_current * k_current).to(matrix_dtype).to(tl.float32)
            log_pi_current = tl.sum(tl.where(row[:, None] == current, log_pi_i, 0.0), 0)
            row_ratio = tl.exp(tl.where(row[:, None] >= current, log_pi_i - log_pi_current[None, :], 0.0))
            column_ratio = tl.exp(tl.where(row[:, None] <= current, log_pi_current[None, :] - log_pi_i, 0.0))
            dtril_row = tl.load(dtril_Q_pi_bar_K_T + (chunk * 32 + row) * 32 + current).to(tl.float32) * SCALE
            dT_L_row = tl.load(dT_L + (chunk * 32 + row) * 32 + current).to(tl.float32)
            dtril_column = tl.load(dtril_Q_pi_bar_K_T + (chunk * 32 + current) * 32 + row).to(tl.float32) * SCALE
            dT_L_column = tl.load(dT_L + (chunk * 32 + current) * 32 + row).to(tl.float32)
            dq_value += tl.where(
                row[:, None] >= current,
                dtril_row[:, None] * row_ratio * k_current[None, :],
                0.0,
            )
            dbeta_k_value += tl.where(
                row[:, None] > current,
                dT_L_row[:, None] * row_ratio * k_current[None, :],
                0.0,
            )
            dk_value += column_ratio * (
                tl.where(row[:, None] <= current, dtril_column[:, None] * q_current[None, :], 0.0)
                + tl.where(row[:, None] < current, dT_L_column[:, None] * beta_k_current[None, :], 0.0)
            )

    dlog_pi_i = dq_value * q_value + dbeta_k_value * beta_k_value - dk_value * k_value
    log_pi_C_value = tl.load(log_pi_C + chunk * DK + key, key < DK, 0).to(tl.float32)
    dQ_pi_value = tl.load(dQ_pi + packed, packed_mask, 0).to(tl.float32) * tl.exp(log_pi_i) * SCALE
    dbar_U_value = tl.load(dbar_U + packed, packed_mask, 0).to(tl.float32) * tl.exp(log_pi_i)
    dbar_K_Pi_C_value = tl.load(dbar_K_Pi_C + packed, packed_mask, 0).to(tl.float32) * tl.exp(
        log_pi_C_value[None, :] - log_pi_i
    )
    dq_value += dQ_pi_value
    dbeta_k_value += dbar_U_value
    dk_value += dbar_K_Pi_C_value
    dlog_pi_i += dQ_pi_value * q_value + dbar_U_value * beta_k_value - dbar_K_Pi_C_value * k_value
    dlog_pi_C_value = tl.load(dlog_pi_C + chunk * DK + key, key < DK, 0).to(tl.float32)
    dlog_pi_C_value += tl.sum(dbar_K_Pi_C_value * k_value, 0)
    dlog_alpha_value = tl.cumsum(dlog_pi_i, 0, reverse=True) + dlog_pi_C_value[None, :]
    tl.store(dq_normalized + packed, dq_value, packed_mask)
    tl.store(dk_normalized + packed, dk_value, packed_mask)
    tl.store(dbeta_k + packed, dbeta_k_value, packed_mask)
    tl.store(dlog_alpha + source, dlog_alpha_value, source_mask)


def left_preparation_bwd(
    q_normalized, k_normalized, beta, log_alpha, log_pi_C,
    dT_L, dtril_Q_pi_bar_K_T, dQ_pi, dbar_U, dbar_K_Pi_C, dlog_pi_C,
    layout, scale, log_alpha_dtype, matrix_dtype,
):
    dq_normalized = torch.empty(*layout.shape, q_normalized.shape[-1], device=q_normalized.device, dtype=matrix_dtype)
    dk_normalized = torch.empty_like(dq_normalized)
    dbeta_k = torch.empty_like(dq_normalized)
    dlog_alpha = torch.empty(
        layout.batch, layout.length, layout.heads, q_normalized.shape[-1],
        device=q_normalized.device, dtype=log_alpha_dtype,
    )
    dk_dim = q_normalized.shape[-1]
    ddn_left_preparation_bwd_kernel[(triton.cdiv(dk_dim, 32) * layout.chunks * layout.heads,)](
        q_normalized, k_normalized, beta, log_alpha, log_pi_C,
        dT_L, dtril_Q_pi_bar_K_T, dQ_pi, dbar_U, dbar_K_Pi_C, dlog_pi_C,
        dq_normalized, dk_normalized, dbeta_k, dlog_alpha,
        layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dk_dim, 32,
        layout.cu_seqlens is not None, scale, matrix_dtype == torch.bfloat16,
        num_warps=4,
        num_stages=1,
    )
    return dq_normalized, dk_normalized, dbeta_k, dlog_alpha
