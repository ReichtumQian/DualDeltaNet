# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def ddn_causal_readout_fwd_kernel(
    Q_eff, S_n, Phi, F_R, hat_V,
    O, O_base, G_R,
    cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    PACKED: tl.constexpr, SAVE_FOR_BACKWARD: tl.constexpr, OUTPUT: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, 32)
    value = tl.arange(0, BV)
    O_base_value = tl.zeros((32, BV), tl.float32)

    for block in range(tl.cdiv(DK, 32)):
        key = block * 32 + row
        Q_eff_value = tl.load(
            Q_eff + (chunk * 32 + row[:, None]) * DK + key[None, :],
            key[None, :] < DK,
            0,
        ).to(hat_V.dtype.element_ty)
        S_n_value = tl.load(
            S_n + (chunk * DK + key[:, None]) * DV + value[None, :],
            (key[:, None] < DK) & (value[None, :] < DV),
            0,
        ).to(hat_V.dtype.element_ty)
        O_base_value += tl.dot(Q_eff_value, S_n_value, input_precision='ieee')

    value_offset = (chunk * 32 + row[:, None]) * DV + value[None, :]
    value_mask = value[None, :] < DV
    F_R_value = tl.load(F_R + value_offset, value_mask, 0)
    matrix_offset = chunk * 1024 + row[:, None] * 32 + row[None, :]
    Phi_value = tl.load(Phi + matrix_offset).to(tl.float32)
    G_R_value = Phi_value - tl.dot(
        O_base_value.to(F_R_value.dtype), tl.trans(F_R_value), input_precision='ieee'
    )
    G_R_value = tl.where(row[:, None] >= row[None, :], G_R_value, 0)

    if SAVE_FOR_BACKWARD:
        tl.store(O_base + value_offset, O_base_value, value_mask)
        tl.store(G_R + matrix_offset, G_R_value)
    if OUTPUT:
        hat_V_value = tl.load(hat_V + value_offset, value_mask, 0)
        O_value = O_base_value + tl.dot(
            G_R_value.to(hat_V_value.dtype), hat_V_value, input_precision='ieee'
        )
        head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
        token = bos + row
        tl.store(
            O + (token[:, None] * H + head) * DV + value[None, :],
            O_value,
            (token[:, None] < eos) & value_mask,
        )


def causal_readout_fwd(
    Q_eff, S_n, Phi, F_R, hat_V, layout, dtype, save_for_backward, output=True,
):
    dk, dv = Q_eff.shape[-1], hat_V.shape[-1]
    O = torch.empty(
        layout.batch, layout.length, layout.heads, dv,
        device=hat_V.device, dtype=dtype,
    ) if output else None
    O_base = torch.empty_like(hat_V) if save_for_backward else None
    G_R = torch.empty_like(Phi) if save_for_backward else None
    ddn_causal_readout_fwd_kernel[(layout.chunks * layout.heads,)](
        Q_eff, S_n, Phi, F_R, hat_V,
        O,
        O_base, G_R,
        layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dk, dv,
        max(16, triton.next_power_of_2(dv)),
        layout.cu_seqlens is not None, save_for_backward, output,
        num_warps=4, num_stages=1,
    )
    return O, O_base, G_R
