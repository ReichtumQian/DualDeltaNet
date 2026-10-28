# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl


@triton.jit
def ddn_left_operator_bwd_kernel(
    M_L, G_L, bar_U, bar_K_Pi_C, tril_Q_pi_bar_K_T,
    dQ_eff, dG_L_in, dPi_C_H, dbar_U_in,
    dbar_U, dbar_K_Pi_C, dtril_Q_pi_bar_K_T, dT_L,
    DK: tl.constexpr, PRECISION: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, 32)
    key_row = tl.arange(0, 32)
    matrix = chunk * 1024 + row[:, None] * 32 + row[None, :]
    M_L_value = tl.load(M_L + matrix).to(bar_U.dtype.element_ty).to(tl.float32)
    G_L_value = tl.load(G_L + matrix).to(tl.float32)
    dG_L_correction = tl.zeros((32, 32), tl.float32)
    dM_L_value = tl.zeros((32, 32), tl.float32)
    for block in range(tl.cdiv(DK, 32)):
        key = block * 32 + key_row
        packed_key = (chunk * 32 + row[:, None]) * DK + key[None, :]
        key_mask = key[None, :] < DK
        Pi_C_H_key = (chunk * DK + key[:, None]) * 32 + row[None, :]
        Pi_C_H_mask = key[:, None] < DK
        bar_U_value = tl.load(bar_U + packed_key, key_mask, 0).to(tl.float32)
        dQ_eff_value = tl.load(dQ_eff + packed_key, key_mask, 0).to(tl.float32)
        dG_L_correction = tl.dot(
            dQ_eff_value, tl.trans(bar_U_value), dG_L_correction, input_precision=PRECISION,
        )
        dbar_U_value = tl.load(dbar_U_in + packed_key, key_mask, 0).to(tl.float32)
        dbar_U_value -= tl.dot(tl.trans(G_L_value), dQ_eff_value, input_precision=PRECISION)
        dPi_C_H_value = tl.load(dPi_C_H + Pi_C_H_key, Pi_C_H_mask, 0).to(tl.float32)
        bar_K_Pi_C_value = tl.load(bar_K_Pi_C + packed_key, key_mask, 0).to(tl.float32)
        dM_L_value = tl.dot(bar_K_Pi_C_value, dPi_C_H_value, dM_L_value, input_precision=PRECISION)
        dbar_K_Pi_C_value = tl.dot(M_L_value, tl.trans(dPi_C_H_value), input_precision=PRECISION)
        tl.store(dbar_U + packed_key, dbar_U_value, key_mask)
        tl.store(dbar_K_Pi_C + packed_key, dbar_K_Pi_C_value, key_mask)
    dG_L_value = tl.load(dG_L_in + matrix).to(tl.float32) - dG_L_correction
    tril_Q_pi_bar_K_T_value = tl.load(tril_Q_pi_bar_K_T + matrix).to(bar_U.dtype.element_ty).to(tl.float32)
    dM_L_value += tl.dot(
        tl.trans(tril_Q_pi_bar_K_T_value), dG_L_value, input_precision=PRECISION,
    )
    dtril_value = tl.dot(
        dG_L_value, tl.trans(M_L_value), input_precision=PRECISION,
    )
    dT_L_value = -tl.dot(
        tl.dot(tl.trans(M_L_value), dM_L_value, input_precision=PRECISION),
        tl.trans(M_L_value),
        input_precision=PRECISION,
    )
    dT_L_value = tl.where(row[:, None] > row[None, :], dT_L_value, 0.0)
    tl.store(dtril_Q_pi_bar_K_T + matrix, dtril_value)
    tl.store(dT_L + matrix, dT_L_value)


def left_operator_bwd(
    M_L, G_L, bar_U, bar_K_Pi_C, tril_Q_pi_bar_K_T,
    dQ_eff, dG_L, dPi_C_H, dbar_U,
):
    outputs = (
        dQ_eff,
        torch.empty_like(bar_U),
        torch.empty_like(bar_K_Pi_C),
        torch.empty(M_L.shape, device=M_L.device, dtype=bar_U.dtype),
        torch.empty(M_L.shape, device=M_L.device, dtype=bar_U.dtype),
    )
    dk = bar_U.shape[-1]
    ddn_left_operator_bwd_kernel[(M_L.numel() // 1024,)](
        M_L, G_L, bar_U, bar_K_Pi_C, tril_Q_pi_bar_K_T,
        dQ_eff, dG_L, dPi_C_H, dbar_U,
        *outputs[1:],
        dk,
        'tf32' if bar_U.dtype == torch.bfloat16 else 'ieee',
        num_warps=4,
        num_stages=1,
    )
    return outputs
