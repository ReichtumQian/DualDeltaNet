# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.ddn import chunk_ddn, fused_recurrent_ddn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class DDN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-6,
        gamma_init: float = 0.05,
        transition_eps: float = 1e-6,
        chunk_size: int = 32,
        disable_recompute: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.layer_idx = layer_idx

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} does not produce an integer value head dimension.")
        if not 1 <= self.head_k_dim <= 128 or not 1 <= self.head_v_dim <= 128:
            raise ValueError("DDN requires key and value head dimensions in [1, 128].")
        if self.num_v_heads < self.num_heads or self.num_v_heads % self.num_heads != 0:
            raise ValueError("`num_v_heads` must be a multiple of `num_heads` and at least as large.")
        if mode not in ("chunk", "fused_recurrent"):
            raise ValueError(f"Unsupported mode `{mode}`.")
        if not 0 < gamma_init < 1:
            raise ValueError(f"`gamma_init` must be between 0 and 1, got {gamma_init}.")
        if not math.isfinite(transition_eps) or transition_eps <= 0:
            raise ValueError(f"`transition_eps` must be positive, got {transition_eps}.")
        if chunk_size != 32:
            raise ValueError(f"`chunk_size` must be 32 for DDN, got {chunk_size}.")
        self.gamma_init = gamma_init
        self.transition_eps = transition_eps
        self.chunk_size = chunk_size
        self.disable_recompute = disable_recompute

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)

        self.gate_dim = self.num_v_heads * self.head_k_dim
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.direction_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=False),
        )
        self.direction_proj[-1]._is_ddn_direction_proj = True
        nn.init.zeros_(self.direction_proj[-1].weight)
        self.gate_proj = nn.Linear(hidden_size, 2 * self.num_v_heads, bias=False)
        self.gate_proj._is_ddn_gate_proj = True
        nn.init.zeros_(self.gate_proj.weight[self.num_v_heads:])
        self.gamma_bias = nn.Parameter(torch.full((self.num_v_heads,), math.log(gamma_init / (1 - gamma_init))))
        self.gamma_bias._no_weight_decay = True
        self.alpha_log_rate = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
        self.alpha_log_rate._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if kwargs.get("cp_context") is not None:
            raise NotImplementedError("DDN context parallelism is not supported.")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("`attention_mask` must have shape `[batch_size, sequence_length]`.")

        batch_size, q_len, _ = hidden_states.shape
        use_recurrent = q_len <= 64 and not self.training and not torch.is_grad_enabled()
        mode = "fused_recurrent" if use_recurrent else self.mode
        if self.training and mode != "chunk":
            raise ValueError("DDN training requires chunk mode.")

        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        cu_seqlens_cpu = kwargs.get("cu_seqlens_cpu")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        conv_states = [None, None, None]
        if last_state is not None and self.use_short_conv:
            conv_states = list(last_state["conv_state"])
        if self.use_short_conv:
            projected = (self.q_proj, self.k_proj, self.v_proj)
            convolutions = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            tensors = []
            for i, (projection, convolution) in enumerate(zip(projected, convolutions)):
                projected_input = projection(hidden_states)
                tensor, conv_states[i] = convolution(
                    x=projected_input,
                    cache=conv_states[i],
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
                tensors.append(tensor)
            q, k, v = tensors
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        dt_logits = self.f_proj(hidden_states)
        gate_logits = F.linear(hidden_states, self.gate_proj.weight, F.pad(self.gamma_bias, (self.num_v_heads, 0)))
        beta_logits, gamma_logits = gate_logits.chunk(2, dim=-1)
        q, k, dt_logits = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, dt_logits))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k)
            )

        direction_logits = rearrange(self.direction_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        gamma_logits = rearrange(gamma_logits, "... h -> ... h 1")
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_ddn(
                q=q,
                k=k,
                v=v,
                dt_logits=dt_logits,
                beta_logits=beta_logits,
                gamma_logits=gamma_logits,
                direction_logits=direction_logits,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                transition_eps=self.transition_eps,
                chunk_size=self.chunk_size,
                alpha_log_rate=self.alpha_log_rate,
                dt_bias=self.dt_bias,
                disable_recompute=self.disable_recompute,
            )
        else:
            o, recurrent_state = fused_recurrent_ddn(
                q=q,
                k=k,
                v=v,
                dt_logits=dt_logits,
                alpha_log_rate=self.alpha_log_rate,
                dt_bias=self.dt_bias,
                beta_logits=beta_logits,
                gamma_logits=gamma_logits,
                direction_logits=direction_logits,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                inplace_final_state=bool(use_cache) and q_len == 1 and recurrent_state is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                transition_eps=self.transition_eps,
                chunk_size=self.chunk_size,
            )
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=tuple(conv_states) if self.use_short_conv else None,
            offset=q_len,
        )
        gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        o = self.o_norm(o, gate)
        o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        o = repad_hidden_states(o, indices, batch_size, q_len)
        return o, None, past_key_values
