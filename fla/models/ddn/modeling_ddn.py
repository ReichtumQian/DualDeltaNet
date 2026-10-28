# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import math

import torch
import torch.nn as nn

from fla.layers.ddn import DDN
from fla.layers.kda import KimiDeltaAttention
from fla.models.ddn.configuration_ddn import DDNConfig
from fla.models.kda.modeling_kda import (
    KDABlock,
    KDAForCausalLM,
    KDAModel,
    KDAPreTrainedModel,
)
from fla.modules import RMSNorm


class DDNBlock(KDABlock):

    def __init__(self, config: DDNConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if isinstance(self.attn, KimiDeltaAttention):
            self.attn = DDN(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                gamma_init=config.gamma_init,
                transition_eps=config.transition_eps,
                chunk_size=config.chunk_size,
                disable_recompute=config.disable_recompute,
                layer_idx=layer_idx,
            )


class _DDNInitMixin:

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, nn.Linear) and getattr(module, "_is_ddn_direction_proj", False):
            nn.init.zeros_(module.weight)
            return
        if isinstance(module, nn.Linear) and getattr(module, "_is_ddn_gate_proj", False):
            heads = module.out_features // 2
            nn.init.normal_(module.weight[:heads], mean=0.0, std=self.config.initializer_range)
            nn.init.zeros_(module.weight[heads:])
            return
        if isinstance(module, DDN) and next(module.parameters()).device.type != "meta":
            with torch.no_grad():
                if not getattr(module.gamma_bias, "_is_hf_initialized", False):
                    nn.init.constant_(module.gamma_bias, math.log(module.gamma_init / (1 - module.gamma_init)))
                module.gamma_bias._no_weight_decay = True
                if not getattr(module.alpha_log_rate, "_is_hf_initialized", False):
                    module.alpha_log_rate.copy_(nn.init.uniform_(module.alpha_log_rate, a=1, b=16).log())
                module.alpha_log_rate._no_weight_decay = True
                if not getattr(module.dt_bias, "_is_hf_initialized", False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                    ).clamp(min=1e-4)
                    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                module.dt_bias._no_weight_decay = True
        super()._init_weights(module, prenorm_residual_strategy, num_residuals_per_layer)


class DDNModel(_DDNInitMixin, KDAModel):
    config_class = DDNConfig
    _no_split_modules = ["DDNBlock"]

    def __init__(self, config: DDNConfig):
        KDAPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DDNBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True
        self.gradient_checkpointing = False
        self.post_init()


class DDNForCausalLM(_DDNInitMixin, KDAForCausalLM):
    config_class = DDNConfig
    _no_split_modules = ["DDNBlock"]

    def __init__(self, config: DDNConfig):
        KDAPreTrainedModel.__init__(self, config)
        self.model = DDNModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()


__all__ = ["DDNBlock", "DDNForCausalLM", "DDNModel"]
