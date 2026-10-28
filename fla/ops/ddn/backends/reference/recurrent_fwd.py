# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import torch


def recurrent_fwd(q, k, beta_k, gamma_e, hat_v, payload, log_alpha, scale, initial_state=None, cu_seqlens=None):
    """Evaluate (I - k beta_k^T) D S (I - gamma_e hat_v^T) + k payload^T token by token."""
    batch, length, heads, key_dim = q.shape
    value_dim = hat_v.shape[-1]
    if cu_seqlens is None:
        spans = [(i, 0, length) for i in range(batch)]
    else:
        offsets = cu_seqlens.tolist()
        spans = [(0, start, end) for start, end in zip(offsets[:-1], offsets[1:])]
    outputs, states = [], []
    for seq, (batch_id, start, end) in enumerate(spans):
        state = q.new_zeros(heads, key_dim, value_dim) if initial_state is None else initial_state[seq].to(q)
        tokens = []
        for t in range(start, end):
            decayed = log_alpha[batch_id, t].to(q).exp().unsqueeze(-1) * state
            left = decayed - k[batch_id, t, :, :, None] * torch.einsum(
                'hk,hkv->hv', beta_k[batch_id, t], decayed,
            )[:, None, :]
            write = payload[batch_id, t]
            state = left - torch.einsum('hkv,hv->hk', left, gamma_e[batch_id, t])[:, :, None] * hat_v[
                batch_id, t, :, None, :
            ] + k[batch_id, t, :, :, None] * write[:, None, :]
            tokens.append(torch.einsum('hk,hkv->hv', q[batch_id, t] * scale, state))
        outputs.append(torch.stack(tokens) if tokens else q.new_empty(0, heads, value_dim))
        states.append(state)
    out = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    return out, torch.stack(states)
