# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.ddn.layout import chunk_bounds


@triton.jit
def ddn_token_value_bwd_kernel(
    v_rstd, m_logits, beta, gamma, hat_V,
    dE, dhat_V, deta, dgamma, dbeta,
    dv, dm_logits, dbeta_logits, dgamma_logits,
    cu_seqlens, indices,
    H: tl.constexpr, T: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    PACKED: tl.constexpr, EPS: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head, bos, eos = chunk_bounds(chunk, cu_seqlens, indices, H, T, PACKED)
    row = tl.arange(0, 32)
    value = tl.arange(0, BV)
    token = bos + row
    source = (token[:, None] * H + head) * DV + value[None, :]
    source_mask = (token[:, None] < eos) & (value[None, :] < DV)
    packed = (chunk * 32 + row[:, None]) * DV + value[None, :]
    packed_mask = value[None, :] < DV

    m_logits_value = tl.load(m_logits + source, source_mask, 0).to(tl.float32)
    m = tl.sigmoid(m_logits_value)
    hat_V_value = tl.load(hat_V + packed, packed_mask, 0).to(tl.float32)
    m_hat_V = m * hat_V_value
    zeta_rstd = tl.rsqrt(tl.sum(m_hat_V * m_hat_V, 1) + EPS)
    E_value = m_hat_V * zeta_rstd[:, None]
    dE_value = tl.load(dE + packed, packed_mask, 0).to(tl.float32)
    dhat_V_value = tl.load(dhat_V + packed, packed_mask, 0).to(tl.float32)
    deta_value = tl.load(deta + chunk * 32 + row).to(tl.float32)
    dgamma_value = tl.load(dgamma + chunk * 32 + row).to(tl.float32)
    dbeta_value = tl.load(dbeta + token * H + head, token < eos, 0).to(tl.float32)
    beta_value = tl.load(beta + token * H + head, token < eos, 0).to(tl.float32)
    gamma_value = tl.load(gamma + chunk * 32 + row).to(tl.float32)
    nu_rstd = tl.load(v_rstd + chunk * 32 + row).to(tl.float32)
    nu = 1.0 / nu_rstd
    v_value = hat_V_value * nu[:, None]
    tau = tl.sum(v_value * E_value, 1)
    dtau = deta_value * gamma_value * (1.0 - beta_value)
    dbeta_value += deta_value * (nu - gamma_value * tau)
    dgamma_value += deta_value * (1.0 - beta_value) * tau
    dv_value = dtau[:, None] * E_value
    dE_value += dtau[:, None] * v_value

    dw = zeta_rstd[:, None] * (
        dE_value - E_value * tl.sum(E_value * dE_value, 1)[:, None]
    )
    dm_logits_value = dw * hat_V_value * m * (1.0 - m)
    dhat_V_value += dw * m

    dv_value += deta_value[:, None] * beta_value[:, None] * hat_V_value
    dv_value += nu_rstd[:, None] * (
        dhat_V_value - hat_V_value * tl.sum(hat_V_value * dhat_V_value, 1)[:, None]
    )
    tl.store(dv + source, dv_value, source_mask)
    tl.store(dm_logits + source, dm_logits_value, source_mask)
    tl.store(dbeta_logits + token * H + head, dbeta_value * beta_value * (1.0 - beta_value), token < eos)
    tl.store(
        dgamma_logits + token * H + head,
        dgamma_value * gamma_value * (1.0 - gamma_value),
        token < eos,
    )


def token_value_bwd(
    v_rstd, m_logits, beta, gamma, hat_V,
    dE, dhat_V, deta, dgamma, dbeta, layout, eps, beta_logits_dtype, gamma_logits_shape, gamma_logits_dtype,
):
    dv_dim = hat_V.shape[-1]
    dv = torch.empty(layout.batch, layout.length, layout.heads, dv_dim, device=hat_V.device, dtype=torch.bfloat16)
    dm_logits = torch.empty_like(m_logits)
    dbeta_logits = torch.empty_like(beta, dtype=beta_logits_dtype)
    dgamma_logits = torch.empty(gamma_logits_shape, device=hat_V.device, dtype=gamma_logits_dtype)
    ddn_token_value_bwd_kernel[(layout.chunks * layout.heads,)](
        v_rstd, m_logits, beta, gamma, hat_V,
        dE, dhat_V, deta, dgamma, dbeta,
        dv, dm_logits, dbeta_logits, dgamma_logits,
        layout.cu_seqlens, layout.indices,
        layout.heads, layout.length, dv_dim, max(16, triton.next_power_of_2(dv_dim)),
        layout.cu_seqlens is not None, eps,
        num_warps=4,
    )
    return dv, dm_logits, dbeta_logits, dgamma_logits
