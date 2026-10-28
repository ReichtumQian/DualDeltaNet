# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def ddn_causal_output_bwd_kernel(
    dO, O_base, G_R, F_R, hat_V,
    dO_base, dPhi, dF_R, dhat_V,
    cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    PACKED: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    row = tl.arange(0, 32)
    value = tl.arange(0, BV)
    token = bos + row
    packed_value = (chunk * 32 + row[:, None]) * DV + value[None, :]
    value_mask = value[None, :] < DV
    dO_value = tl.load(
        dO + (token[:, None] * H + head) * DV + value[None, :],
        (token[:, None] < eos) & value_mask,
        0,
    )
    hat_V_value = tl.load(hat_V + packed_value, value_mask, 0)
    dG_R = tl.dot(
        dO_value.to(hat_V_value.dtype), tl.trans(hat_V_value), input_precision='ieee',
    )
    dG_R = tl.where(row[:, None] >= row[None, :], dG_R, 0.0)

    matrix = chunk * 1024 + row[:, None] * 32 + row[None, :]
    tl.store(dPhi + matrix, dG_R)
    F_R_value = tl.load(F_R + packed_value, value_mask, 0)
    dO_base_value = dO_value - tl.dot(
        dG_R.to(F_R_value.dtype), F_R_value, input_precision='ieee',
    )
    tl.store(dO_base + packed_value, dO_base_value, value_mask)
    O_base_value = tl.load(O_base + packed_value, value_mask, 0)
    dF_R_value = -tl.dot(
        tl.trans(dG_R).to(O_base_value.dtype), O_base_value, input_precision='ieee',
    )
    tl.store(dF_R + packed_value, dF_R_value, value_mask)
    G_R_value = tl.load(G_R + matrix)
    dhat_V_value = tl.dot(
        tl.trans(G_R_value), dO_value.to(G_R_value.dtype), input_precision='ieee',
    )
    tl.store(dhat_V + packed_value, dhat_V_value, value_mask)


def causal_output_bwd(dO, O_base, G_R, F_R, hat_V, layout):
    dO_base = torch.empty_like(hat_V)
    dPhi = torch.empty_like(G_R, dtype=hat_V.dtype)
    dF_R = torch.empty_like(F_R)
    dhat_V = torch.empty_like(hat_V)
    dv = hat_V.shape[-1]
    ddn_causal_output_bwd_kernel[(layout.chunks * layout.heads,)](
        dO, O_base, G_R, F_R, hat_V,
        dO_base, dPhi, dF_R, dhat_V,
        layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dv, max(16, triton.next_power_of_2(dv)),
        layout.cu_seqlens is not None,
        num_warps=4,
        num_stages=1,
    )
    return dO_base, dPhi, dF_R, dhat_V
