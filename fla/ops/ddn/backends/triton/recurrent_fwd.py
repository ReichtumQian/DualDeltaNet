# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[triton.Config({}, num_warps=warps, num_stages=1) for warps in (4, 8)],
    key=['N', 'T', 'H', 'DK', 'DV', 'IS_VARLEN', 'HAS_INITIAL_STATE',
         'OUTPUT_FINAL_STATE', 'NORMALIZE_QK', 'INPLACE_FINAL_STATE'],
    restore_value=['InitialState'],
    **autotune_cache_kwargs,
)
@triton.jit
def ddn_recurrent_fwd(Q, K, V, DtLogits, BetaLogits, GammaLogits, DirectionLogits, AlphaLogRate, DtBias,
                           InitialState, FinalState, O, CuSeqlens,
                           BETA_B: tl.constexpr, BETA_T: tl.constexpr, BETA_H: tl.constexpr,
                           GAMMA_B: tl.constexpr, GAMMA_T: tl.constexpr, GAMMA_H: tl.constexpr,
                           N: tl.constexpr, T: tl.constexpr, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                           BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr,
                           HAS_INITIAL_STATE: tl.constexpr, OUTPUT_FINAL_STATE: tl.constexpr,
                           NORMALIZE_QK: tl.constexpr, SCALE: tl.constexpr, TRANSITION_EPS: tl.constexpr,
                           INPLACE_FINAL_STATE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    sequence_idx, head_idx = program_id // H, program_id % H
    if IS_VARLEN:
        sequence_start = tl.load(CuSeqlens + sequence_idx).to(tl.int64)
        sequence_end = tl.load(CuSeqlens + sequence_idx + 1).to(tl.int64)
    else:
        sequence_start, sequence_end = sequence_idx * T, (sequence_idx + 1) * T
    key_indices, value_indices = tl.arange(0, BK), tl.arange(0, BV)
    # Keep the register view [V, K] and the external state layout [K, V].
    state_mask = (value_indices[:, None] < DV) & (key_indices[None, :] < DK)
    state_offset = program_id * DK * DV + key_indices[None, :] * DV + value_indices[:, None]
    state = tl.zeros((BV, BK), tl.float32)
    if HAS_INITIAL_STATE:
        state = tl.load(InitialState + state_offset, state_mask, 0).to(tl.float32)
    alpha_rate = exp(tl.load(AlphaLogRate + head_idx).to(tl.float32))
    for token in range(sequence_start, sequence_end):
        key_offset = (token * H + head_idx) * DK + key_indices
        value_offset = (token * H + head_idx) * DV + value_indices
        dt_logits = tl.load(DtLogits + key_offset, key_indices < DK, 0).to(tl.float32)
        dt_logits += tl.load(DtBias + head_idx * DK + key_indices, key_indices < DK, 0).to(tl.float32)
        log_alpha = (-alpha_rate * softplus(dt_logits)).to(Q.dtype.element_ty).to(tl.float32)
        q = tl.load(Q + key_offset, key_indices < DK, 0).to(tl.float32)
        k = tl.load(K + key_offset, key_indices < DK, 0).to(tl.float32)
        beta_offset = (token // T) * BETA_B + (token % T) * BETA_T + head_idx * BETA_H
        gamma_offset = (token // T) * GAMMA_B + (token % T) * GAMMA_T + head_idx * GAMMA_H
        beta = tl.sigmoid(tl.load(BetaLogits + beta_offset).to(tl.float32))
        if NORMALIZE_QK:
            q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
            k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        v = tl.load(V + value_offset, value_indices < DV, 0).to(tl.float32)
        gamma = tl.sigmoid(tl.load(GammaLogits + gamma_offset).to(tl.float32))
        hat_v = v * tl.rsqrt(tl.sum(v * v, 0) + TRANSITION_EPS)
        direction_logits = tl.load(DirectionLogits + value_offset, value_indices < DV, 0).to(tl.float32)
        m = tl.sigmoid(direction_logits)
        weighted_v = hat_v * m
        e = weighted_v * tl.rsqrt(tl.sum(weighted_v * weighted_v, 0) + TRANSITION_EPS)
        state *= tl.exp(log_alpha)[None, :]
        ell = tl.sum(k[None, :] * state, 1)
        r = tl.sum(state * e[:, None], 0)
        tau = tl.sum(v * e, 0)
        tilde_v = v - ell
        tilde_k = (tau - beta * tl.sum(tilde_v * e, 0)) * k - r
        value_factor = beta * tilde_v
        direction_factor = gamma * hat_v
        state += value_factor[:, None] * k[None, :]
        state += direction_factor[:, None] * tilde_k[None, :]
        output = tl.sum((q * SCALE)[None, :] * state, 1)
        tl.store(O + value_offset, output, value_indices < DV)
    if OUTPUT_FINAL_STATE or INPLACE_FINAL_STATE:
        tl.store(FinalState + state_offset, state, state_mask)


def recurrent_fwd(q, k, v, dt_logits, beta_logits, gamma_logits, alpha_log_rate, dt_bias,
                  initial_state, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel,
                  scale, transition_eps, direction_logits, inplace_final_state=False):
    batch_size, seq_length, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    num_sequences = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    output = torch.empty_like(v)
    if inplace_final_state:
        final_state = initial_state
    elif output_final_state:
        final_state = torch.empty(num_sequences, num_heads, head_k_dim, head_v_dim, device=q.device, dtype=torch.float32)
    else:
        final_state = None
    ddn_recurrent_fwd[(num_sequences * num_heads,)](
        q, k, v, dt_logits, beta_logits, gamma_logits, direction_logits, alpha_log_rate, dt_bias,
        initial_state, final_state, output, cu_seqlens,
        *beta_logits.stride()[:3], *gamma_logits.stride()[:3],
        num_sequences, seq_length, num_heads, head_k_dim, head_v_dim,
        max(16, triton.next_power_of_2(head_k_dim)), max(16, triton.next_power_of_2(head_v_dim)), cu_seqlens is not None,
        initial_state is not None, output_final_state, use_qk_l2norm_in_kernel, scale,
        transition_eps, inplace_final_state,
    )
    return output, final_state
