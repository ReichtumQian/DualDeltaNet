# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import triton
import triton.language as tl


@triton.jit
def solve_tril(T_00, T_10, T_11, PRECISION: tl.constexpr):
    """Return the three 16x16 blocks of (I + T)^-1 for strictly lower-triangular T."""
    row = tl.arange(0, 16)
    M_00, M_11 = -T_00, -T_11
    for current in range(2, 16):
        minus_T_row00 = tl.sum(tl.where(row[:, None] == current, -T_00, 0), 0)
        minus_T_row11 = tl.sum(tl.where(row[:, None] == current, -T_11, 0), 0)
        M_row00 = minus_T_row00 + tl.sum(minus_T_row00[:, None] * M_00, 0)
        M_row11 = minus_T_row11 + tl.sum(minus_T_row11[:, None] * M_11, 0)
        M_00 = tl.where((row == current)[:, None], M_row00[None, :], M_00)
        M_11 = tl.where((row == current)[:, None], M_row11[None, :], M_11)
    I = (row[:, None] == row[None, :]).to(tl.float32)
    M_00 += I
    M_11 += I
    M_10 = -tl.dot(
        tl.dot(M_11, T_10, input_precision=PRECISION),
        M_00,
        input_precision=PRECISION,
    )

    return M_00, M_10, M_11
