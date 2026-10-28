# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable

import torch
from torch import nn
from torchtitan.components.optimizer import FTOptimizersContainer, OptimizersContainer, OptimizersInBackwardContainer


def _group_parameters(parameters: Iterable[nn.Parameter]):
    decay, no_decay = [], []
    for parameter in parameters:
        group = no_decay if getattr(parameter, "_no_weight_decay", False) else decay
        group.append(parameter)

    if not no_decay:
        return decay

    groups = []
    if decay:
        groups.append({"params": decay})
    groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class _Adam(torch.optim.Adam):
    def __init__(self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any):
        super().__init__(_group_parameters(params), *args, **kwargs)


class _AdamW(torch.optim.AdamW):
    def __init__(self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any):
        super().__init__(_group_parameters(params), *args, **kwargs)


def build_optimizers(model_parts, job_config, ft_manager):
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.experimental.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )

    optimizer_classes = {"Adam": _Adam, "AdamW": _AdamW}
    name = job_config.optimizer.name
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")

    implementation = job_config.optimizer.implementation
    assert implementation in ["fused", "foreach", "for-loop"]
    optimizer_kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
        "weight_decay": job_config.optimizer.weight_decay,
        "fused": implementation == "fused",
        "foreach": implementation == "foreach",
    }
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    if ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
