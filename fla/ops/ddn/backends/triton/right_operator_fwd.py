# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.backends.triton.right_inverse_fwd import right_inverse_fwd
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=1) for warps in (2, 4)],
    key=['DV'],
    **autotune_cache_kwargs,
)
@triton.jit
def ddn_right_operator_fwd_kernel(
    E, hat_V, gamma, eta, G_L, M_R, N_Z, F_R, Phi,
    DV: tl.constexpr, BV: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    row = tl.arange(0, 32)
    matrix_offset = chunk * 1024 + row[:, None] * 32 + row[None, :]
    m = tl.load(M_R + matrix_offset).to(tl.bfloat16)
    g = tl.load(G_L + matrix_offset).to(tl.float32)
    eta_value = tl.load(eta + chunk * 32 + row).to(tl.float32)
    phi = tl.dot((g * eta_value[None, :]).to(tl.bfloat16), tl.trans(m))
    tl.store(Phi + matrix_offset, phi)

    gamma_value = tl.load(gamma + chunk * 32 + row).to(tl.float32)
    for block in range(tl.cdiv(DV, BV)):
        value = block * BV + tl.arange(0, BV)
        offset = (chunk * 32 + row[None, :]) * DV + value[:, None]
        mask = value[:, None] < DV
        v = tl.load(hat_V + offset, mask, 0).to(tl.bfloat16)
        z = tl.dot(v, m)
        tl.store(N_Z + offset, z * eta_value[None, :], mask)
        e = tl.load(E + offset, mask, 0).to(tl.float32)
        f = tl.dot((e * gamma_value[None, :]).to(tl.bfloat16), tl.trans(m))
        tl.store(F_R + offset, f, mask)


def right_operator_fwd(E, hat_V, gamma, eta, G_L, M_R=None):
    if M_R is None:
        M_R = right_inverse_fwd(E, hat_V, gamma)
    N_Z = torch.empty_like(hat_V)
    F_R = torch.empty_like(hat_V)
    Phi = torch.empty_like(G_L)
    ddn_right_operator_fwd_kernel[(G_L.numel() // 1024,)](
        E, hat_V, gamma, eta, G_L, M_R, N_Z, F_R, Phi, hat_V.shape[-1], 64,
    )
    return M_R, N_Z, F_R, Phi
