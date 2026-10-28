# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl


@triton.jit
def ddn_base_readout_bwd_kernel(
    Q_eff, S_n, dO_base, dQ_eff, dS_n,
    DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64) // tl.cdiv(DK, 32)
    key = (tl.program_id(0).to(tl.int64) % tl.cdiv(DK, 32)) * 32 + tl.arange(0, 32)
    row = tl.arange(0, 32)
    value = tl.arange(0, BV)
    packed_key = (chunk * 32 + row[:, None]) * DK + key[None, :]
    packed_value = (chunk * 32 + row[:, None]) * DV + value[None, :]
    state = chunk * DK * DV + key[:, None] * DV + value[None, :]
    Q_eff_value = tl.load(Q_eff + packed_key, key[None, :] < DK, 0)
    S_n_value = tl.load(
        S_n + state,
        (key[:, None] < DK) & (value[None, :] < DV),
        0,
    )
    dO_base_value = tl.load(dO_base + packed_value, value[None, :] < DV, 0)
    dQ_eff_value = tl.dot(
        dO_base_value.to(Q_eff_value.dtype),
        tl.trans(S_n_value).to(Q_eff_value.dtype),
        input_precision='ieee',
    )
    dS_n_value = tl.dot(
        tl.trans(Q_eff_value),
        dO_base_value.to(Q_eff_value.dtype),
        input_precision='ieee',
    )
    tl.store(dQ_eff + packed_key, dQ_eff_value, key[None, :] < DK)
    tl.store(
        dS_n + state,
        dS_n_value,
        (key[:, None] < DK) & (value[None, :] < DV),
    )


def base_readout_bwd(Q_eff, S_n, dO_base):
    dQ_eff = torch.empty_like(Q_eff)
    dS_n = torch.empty_like(S_n)
    dk, dv = Q_eff.shape[-1], dO_base.shape[-1]
    chunks_heads = Q_eff.numel() // (32 * dk)
    ddn_base_readout_bwd_kernel[(triton.cdiv(dk, 32) * chunks_heads,)](
        Q_eff, S_n, dO_base, dQ_eff, dS_n,
        dk, dv, max(16, triton.next_power_of_2(dv)),
        num_warps=4,
        num_stages=1,
    )
    return dQ_eff, dS_n
