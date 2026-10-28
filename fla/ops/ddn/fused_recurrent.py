# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch

from fla.ops.ddn.backends.triton.recurrent_fwd import recurrent_fwd
from fla.ops.ddn.chunk import validate_inputs
from fla.utils import input_guard


@torch.compiler.disable
@input_guard(no_guard_contiguous=("beta_logits", "gamma_logits", "initial_state"))
def fused_recurrent_ddn(q, k, v, dt_logits, beta_logits, gamma_logits, alpha_log_rate, dt_bias,
                             scale=None, initial_state=None,
                             output_final_state=False, use_qk_l2norm_in_kernel=False,
                             cu_seqlens=None, cu_seqlens_cpu=None,
                             transition_eps=1e-6, chunk_size=32, direction_logits=None,
                             inplace_final_state=False):
    """Inference-only recurrence with FP32 states and fused decay activation.

    Computes log_alpha = -exp(alpha_log_rate) * softplus(dt_logits + dt_bias),
    rounded to BF16 before applying the decay.

    With inplace_final_state=True, updates and returns initial_state even when
    output_final_state=False. The state must be contiguous and use float32.
    """
    if dt_bias is None:
        raise ValueError('dt_bias is required for fused recurrent DDN.')
    if inplace_final_state:
        if initial_state is None:
            raise ValueError('inplace_final_state=True requires initial_state.')
        if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
            raise ValueError('inplace_final_state=True requires a contiguous float32 initial_state.')
    elif initial_state is not None:
        initial_state = initial_state.contiguous()
    validate_inputs(q, k, v, None, beta_logits, gamma_logits, initial_state,
                    cu_seqlens, cu_seqlens_cpu, chunk_size, transition_eps, scale,
                    direction_logits, alpha_log_rate=alpha_log_rate, dt_bias=dt_bias, dt_logits=dt_logits)
    input_tensors = (q, k, v, beta_logits, gamma_logits, initial_state,
                     direction_logits, dt_logits, alpha_log_rate, dt_bias)
    if torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in input_tensors):
        raise RuntimeError('fused_recurrent_ddn is inference-only; use chunk_ddn for gradients.')
    return recurrent_fwd(q, k, v, dt_logits, beta_logits, gamma_logits, alpha_log_rate, dt_bias, initial_state,
                         output_final_state, cu_seqlens, use_qk_l2norm_in_kernel,
                         q.shape[-1] ** -0.5 if scale is None else scale,
                         transition_eps, direction_logits, inplace_final_state)
