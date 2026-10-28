# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gdn2.configuration_gdn2 import GDN2Config
from fla.models.gdn2.modeling_gdn2 import GDN2ForCausalLM, GDN2Model

AutoConfig.register(GDN2Config.model_type, GDN2Config, exist_ok=True)
AutoModel.register(GDN2Config, GDN2Model, exist_ok=True)
AutoModelForCausalLM.register(GDN2Config, GDN2ForCausalLM, exist_ok=True)

__all__ = ["GDN2Config", "GDN2ForCausalLM", "GDN2Model"]
