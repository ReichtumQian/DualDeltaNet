# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch

from fla.ops.ddn.backends.tilelang.boundary_state_bwd import boundary_state_bwd
from fla.ops.ddn.backends.tilelang.boundary_state_fwd import boundary_state_fwd
from fla.ops.ddn.backends.tilelang.right_operator_bwd import right_operator_bwd
from fla.ops.ddn.backends.triton.base_readout_bwd import base_readout_bwd
from fla.ops.ddn.backends.triton.causal_output_bwd import causal_output_bwd
from fla.ops.ddn.backends.triton.causal_readout_fwd import causal_readout_fwd
from fla.ops.ddn.backends.triton.left_operator_bwd import left_operator_bwd
from fla.ops.ddn.backends.triton.left_operator_fwd import left_operator_fwd
from fla.ops.ddn.backends.triton.left_preparation_bwd import left_preparation_bwd
from fla.ops.ddn.backends.triton.right_operator_fwd import right_operator_fwd
from fla.ops.ddn.backends.triton.token_preparation_fwd import token_preparation_fwd
from fla.ops.ddn.backends.triton.token_qk_bwd import token_qk_bwd
from fla.ops.ddn.backends.triton.token_value_bwd import token_value_bwd
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd


class ChunkDDNFunction(torch.autograd.Function):

    @staticmethod
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        decay_input,
        beta_logits,
        gamma_logits,
        m_logits,
        S_0,
        output_final_state,
        layout,
        dtype,
        normalize_qk,
        eps,
        scale,
        disable_recompute,
        alpha_log_rate,
        dt_bias,
    ):
        log_alpha = (
            kda_gate_fwd(decay_input, alpha_log_rate, dt_bias, output_dtype=q.dtype)
            if alpha_log_rate is not None else decay_input
        )
        (
            q_normalized, k_normalized, Q_pi, bar_U, bar_K_Pi_C, beta, gamma, eta,
            log_pi_C, E, hat_V, q_rstd, k_rstd, v_rstd,
        ) = token_preparation_fwd(
            q, k, v, log_alpha, beta_logits, gamma_logits, m_logits, layout, normalize_qk, eps, scale,
        )
        normalization_tensors = (q_normalized, k_normalized, q_rstd, k_rstd, hat_V, v_rstd)
        matrix_dtype = bar_U.dtype
        hat_V = hat_V.to(matrix_dtype)
        save_intermediates = disable_recompute and any(ctx.needs_input_grad)
        M_L, G_L, Pi_C_H, Q_eff, tril_Q_pi_bar_K_T = left_operator_fwd(
            q_normalized, k_normalized, beta, log_alpha, Q_pi, bar_U, bar_K_Pi_C,
            layout, scale, any(ctx.needs_input_grad),
        )
        M_R, N_Z, F_R, Phi = right_operator_fwd(E, hat_V, gamma, eta, G_L)
        S_n, S_N, Psi = boundary_state_fwd(
            bar_U, N_Z, F_R, hat_V, Pi_C_H, log_pi_C,
            S_0, output_final_state, layout, save_intermediates,
        )
        O, O_base, G_R = causal_readout_fwd(
            Q_eff, S_n, Phi, F_R, hat_V, layout, dtype, save_intermediates,
        )
        if not save_intermediates:
            bar_U = bar_K_Pi_C = log_pi_C = None
            G_L = Pi_C_H = Q_eff = None
            E = eta = F_R = None
            S_n = Psi = O_base = G_R = None
        ctx.save_for_backward(
            decay_input, beta, gamma, m_logits,
            *normalization_tensors,
            M_L, M_R, S_0, tril_Q_pi_bar_K_T, alpha_log_rate, dt_bias,
            bar_U, bar_K_Pi_C, log_pi_C, G_L, Pi_C_H, Q_eff,
            E, eta, F_R, S_n, Psi, O_base, G_R,
        )
        ctx.matrix_dtype = matrix_dtype
        ctx.disable_recompute = disable_recompute
        ctx.gamma_logits_shape = gamma_logits.shape
        ctx.gamma_logits_dtype = gamma_logits.dtype
        ctx.layout = layout
        ctx.normalize_qk = normalize_qk
        ctx.scale = scale
        ctx.log_alpha_dtype = log_alpha.dtype
        ctx.beta_logits_dtype = beta_logits.dtype
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        return O, S_N

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dO, dS_N):
        (
            decay_input, beta, gamma, m_logits,
            q_normalized, k_normalized, q_rstd, k_rstd, hat_V, v_rstd,
            M_L, M_R, S_0, tril_Q_pi_bar_K_T, alpha_log_rate, dt_bias,
            bar_U, bar_K_Pi_C, log_pi_C, G_L, Pi_C_H, Q_eff,
            E, eta, F_R, S_n, Psi, O_base, G_R,
        ) = ctx.saved_tensors
        log_alpha = (
            kda_gate_fwd(decay_input, alpha_log_rate, dt_bias, output_dtype=ctx.log_alpha_dtype)
            if alpha_log_rate is not None else decay_input
        )
        layout = ctx.layout
        if ctx.disable_recompute:
            hat_V = hat_V.to(ctx.matrix_dtype)
        else:
            cached_normalization = (q_normalized, k_normalized, hat_V, beta, gamma, q_rstd, k_rstd, v_rstd)
            (
                _, _, Q_pi, bar_U, bar_K_Pi_C, _, _, eta, log_pi_C, E, hat_V, _, _, _,
            ) = token_preparation_fwd(
                q_normalized, k_normalized, hat_V, log_alpha, None, None, m_logits,
                layout, ctx.normalize_qk, ctx.eps, ctx.scale, cached_normalization=cached_normalization,
            )
            hat_V = hat_V.to(ctx.matrix_dtype)
            _, G_L, Pi_C_H, Q_eff, _ = left_operator_fwd(
                q_normalized, k_normalized, beta, log_alpha, Q_pi, bar_U, bar_K_Pi_C,
                layout, ctx.scale, False, M_L=M_L, tril_Q_pi_bar_K_T=tril_Q_pi_bar_K_T,
            )
            _, N_Z, F_R, Phi = right_operator_fwd(E, hat_V, gamma, eta, G_L, M_R=M_R)
            S_n, _, Psi = boundary_state_fwd(
                bar_U, N_Z, F_R, hat_V, Pi_C_H, log_pi_C, S_0, False, layout, True,
            )
            _, O_base, G_R = causal_readout_fwd(
                Q_eff, S_n, Phi, F_R, hat_V, layout, hat_V.dtype, True, output=False,
            )

        if dO is None:
            dO = hat_V.new_zeros(layout.batch, layout.length, layout.heads, hat_V.shape[-1])
        dO_base, dPhi, dF_R, dhat_V = causal_output_bwd(
            dO, O_base, G_R, F_R, hat_V, layout,
        )
        dQ_eff, dS_n_readout = base_readout_bwd(Q_eff, S_n, dO_base)
        dS_0, dPi_C_H, dN_Z, dbar_U, dF_R, dhat_V, dlog_pi_C = boundary_state_bwd(
            S_n, Psi, Pi_C_H, bar_U, F_R, hat_V, log_pi_C,
            dS_n_readout, dS_N, dF_R, dhat_V, S_0, layout,
        )
        dG_L, deta, dgamma, dE, dhat_V = right_operator_bwd(
            M_R, G_L, eta, gamma, E, hat_V,
            dPhi, dF_R, dN_Z, dhat_V,
        )
        (
            dQ_pi, dbar_U, dbar_K_Pi_C, dtril_Q_pi_bar_K_T, dT_L,
        ) = left_operator_bwd(
            M_L, G_L, bar_U, bar_K_Pi_C, tril_Q_pi_bar_K_T,
            dQ_eff, dG_L, dPi_C_H, dbar_U,
        )
        dq_normalized, dk_normalized, dbeta_k, dlog_alpha = left_preparation_bwd(
            q_normalized, k_normalized, beta, log_alpha, log_pi_C,
            dT_L, dtril_Q_pi_bar_K_T, dQ_pi, dbar_U, dbar_K_Pi_C, dlog_pi_C,
            layout, ctx.scale, ctx.log_alpha_dtype, ctx.matrix_dtype,
        )
        dq, dk, dbeta = token_qk_bwd(
            q_normalized, k_normalized, beta, q_rstd, k_rstd,
            dq_normalized, dk_normalized, dbeta_k, layout, ctx.normalize_qk,
        )
        dv, dm_logits, dbeta_logits, dgamma_logits = token_value_bwd(
            v_rstd, m_logits, beta, gamma, hat_V,
            dE, dhat_V, deta, dgamma, dbeta, layout, ctx.eps, ctx.beta_logits_dtype,
            ctx.gamma_logits_shape, ctx.gamma_logits_dtype,
        )
        ddecay_input, dalpha_log_rate, ddt_bias = (
            kda_gate_bwd(decay_input, alpha_log_rate, dt_bias, dlog_alpha) if alpha_log_rate is not None else (dlog_alpha, None, None)
        )
        return (
            dq, dk, dv, ddecay_input, dbeta_logits, dgamma_logits, dm_logits, dS_0,
            None, None, None, None, None, None, None, dalpha_log_rate, ddt_bias,
        )
