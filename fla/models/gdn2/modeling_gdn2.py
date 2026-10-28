# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch
import torch.nn as nn

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gdn2 import GatedDeltaNet2
from fla.models.gated_deltanet.modeling_gated_deltanet import (
    GatedDeltaNetBlock,
    GatedDeltaNetForCausalLM,
    GatedDeltaNetModel,
    GatedDeltaNetPreTrainedModel,
)
from fla.models.gdn2.configuration_gdn2 import GDN2Config
from fla.modules import RMSNorm


class GDN2Block(GatedDeltaNetBlock):

    def __init__(self, config: GDN2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if isinstance(self.attn, GatedDeltaNet):
            self.attn = GatedDeltaNet2(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                use_gate_in_kernel=config.use_gate_in_kernel,
            )


class _GDN2InitMixin:
    # Repeat the FLA layer's special-parameter initialization after meta allocation.

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, GatedDeltaNet2):
            if module.A_log.device.type != "meta":
                with torch.no_grad():
                    if not getattr(module.A_log, "_is_hf_initialized", False):
                        nn.init.uniform_(module.A_log, a=1, b=16)
                        module.A_log.log_()
                    if not getattr(module.dt_bias, "_is_hf_initialized", False):
                        dt = torch.empty_like(module.dt_bias).uniform_(math.log(0.001), math.log(0.1)).exp_()
                        dt.clamp_(min=1e-4)
                        module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            module.A_log._no_weight_decay = True
            module.dt_bias._no_weight_decay = True
            return

        super()._init_weights(module, prenorm_residual_strategy, num_residuals_per_layer)


class GDN2Model(_GDN2InitMixin, GatedDeltaNetModel):
    config_class = GDN2Config
    _no_split_modules = ["GDN2Block"]

    def __init__(self, config: GDN2Config):
        GatedDeltaNetPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GDN2Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = config.attnres_block_size is not None
        if self.use_attnres:
            self.res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.res_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.res_proj._is_attnres_proj = True
        self.gradient_checkpointing = False
        self.post_init()


class GDN2ForCausalLM(_GDN2InitMixin, GatedDeltaNetForCausalLM):
    config_class = GDN2Config
    _no_split_modules = ["GDN2Block"]

    def __init__(self, config: GDN2Config):
        GatedDeltaNetPreTrainedModel.__init__(self, config)
        self.model = GDN2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()


__all__ = ["GDN2Block", "GDN2ForCausalLM", "GDN2Model"]
