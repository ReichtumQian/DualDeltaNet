# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig


class GDN2Config(GatedDeltaNetConfig):
    model_type = "gdn2"

    def __init__(
        self,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        use_gate_in_kernel: bool = False,
        **kwargs,
    ) -> None:
        self.use_gate_in_kernel = use_gate_in_kernel
        super().__init__(expand_v=expand_v, head_dim=head_dim, num_heads=num_heads, **kwargs)


__all__ = ["GDN2Config"]
