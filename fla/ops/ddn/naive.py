# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch

from fla.ops.ddn.backends.reference.recurrent_fwd import recurrent_fwd


def naive_recurrent_ddn(q, k, v, log_alpha, beta_logits, gamma_logits, scale=None, initial_state=None,
                             output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                             transition_eps=1e-6, cu_seqlens_cpu=None, chunk_size=32, direction_logits=None):
    """Differentiable DDN oracle; float64 inputs retain float64 arithmetic for gradcheck."""
    if direction_logits is None:
        raise ValueError('DDN requires direction_logits.')
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qn, kn, values = (x.to(dtype) for x in (q, k, v))
    if use_qk_l2norm_in_kernel:
        qn = qn * (qn.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        kn = kn * (kn.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    hat_v = values * (values.square().sum(-1, keepdim=True) + transition_eps).rsqrt()
    m = direction_logits.to(qn).sigmoid()
    e = hat_v * m
    e = e * (e.square().sum(-1, keepdim=True) + transition_eps).rsqrt()
    tau = (values * e).sum(-1, keepdim=True)
    beta = beta_logits.reshape(*v.shape[:-1], 1).to(qn).sigmoid()
    gamma = gamma_logits.reshape(*v.shape[:-1], 1).to(qn).sigmoid()
    payload = beta * values + gamma * (1 - beta) * tau * hat_v
    out, final = recurrent_fwd(qn, kn, beta * kn, gamma * e, hat_v, payload, log_alpha,
                               scale if scale is not None else q.shape[-1] ** -0.5, initial_state, cu_seqlens)
    return out.to(q.dtype), final if output_final_state else None
