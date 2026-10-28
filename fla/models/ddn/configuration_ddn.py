# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import math

from fla.models.kda.configuration_kda import KDAConfig


class DDNConfig(KDAConfig):
    model_type = "ddn"

    def __init__(
        self,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        gamma_init: float = 0.05,
        transition_eps: float = 1e-6,
        chunk_size: int = 32,
        disable_recompute: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("allow_neg_eigval", False) or kwargs.get("safe_gate", False) or kwargs.get("lower_bound") is not None:
            raise ValueError("DDN supports beta in [0, 1] and the standard KDA decay parameterization.")
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
        super().__init__(expand_v=expand_v, head_dim=head_dim, num_heads=num_heads, **kwargs)


__all__ = ["DDNConfig"]
