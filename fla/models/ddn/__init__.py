# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.ddn.configuration_ddn import DDNConfig
from fla.models.ddn.modeling_ddn import DDNForCausalLM, DDNModel

AutoConfig.register(DDNConfig.model_type, DDNConfig, exist_ok=True)
AutoModel.register(DDNConfig, DDNModel, exist_ok=True)
AutoModelForCausalLM.register(DDNConfig, DDNForCausalLM, exist_ok=True)

__all__ = ["DDNConfig", "DDNForCausalLM", "DDNModel"]
