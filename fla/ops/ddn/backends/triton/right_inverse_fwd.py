# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.backends.triton.solve_tril import solve_tril
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=1) for warps in (2, 4)],
    key=['DV'],
    **autotune_cache_kwargs,
)
@triton.jit
def ddn_right_inverse_fwd_kernel(
    E, hat_V, gamma, M_R,
    DV: tl.constexpr, BV: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, 32)
    value = tl.arange(0, BV)
    offset = (chunk * 32 + row[:, None]) * DV + value[None, :]
    mask = value[None, :] < DV
    e = tl.load(E + offset, mask, 0).to(tl.float32)
    v = tl.load(hat_V + offset, mask, 0).to(tl.bfloat16)
    g = tl.load(gamma + chunk * 32 + row).to(tl.float32)
    interaction = tl.dot((g[:, None] * e).to(tl.bfloat16), tl.trans(v))
    interaction = tl.where(row[:, None] > row[None, :], interaction, 0)

    blocks = tl.permute(tl.reshape(interaction, (2, 16, 2, 16)), (1, 3, 0, 2))
    left_column, right_column = tl.split(blocks)
    T_00, T_10 = tl.split(left_column)
    _, T_11 = tl.split(right_column)
    M_00, M_10, M_11 = solve_tril(T_00, T_10, T_11, 'tf32')
    zero = tl.zeros((16, 16), tl.float32)
    inverse_blocks = tl.join(tl.join(M_00, M_10), tl.join(zero, M_11))
    inverse = tl.reshape(tl.permute(inverse_blocks, (2, 0, 3, 1)), (32, 32))
    tl.store(M_R + chunk * 1024 + row[:, None] * 32 + row[None, :], inverse)


def right_inverse_fwd(E, hat_V, gamma):
    dv = hat_V.shape[-1]
    M_R = torch.empty(*E.shape[:-2], 32, 32, device=E.device, dtype=torch.bfloat16)
    ddn_right_inverse_fwd_kernel[(M_R.numel() // 1024,)](
        E, hat_V, gamma, M_R, dv, max(16, triton.next_power_of_2(dv)),
    )
    return M_R
