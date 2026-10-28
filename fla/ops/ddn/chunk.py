# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import math

import torch

from fla.ops.ddn.core import ChunkDDNFunction
from fla.ops.ddn.layout import make_layout
from fla.utils import input_guard


def validate_inputs(q, k, v, log_alpha, beta_logits, gamma_logits, initial_state,
                    cu_seqlens, cu_seqlens_cpu, chunk_size, transition_eps, scale,
                    direction_logits, alpha_log_rate=None, dt_bias=None, dt_logits=None):
    if (log_alpha is None) == (dt_logits is None):
        raise ValueError('Provide exactly one of log_alpha and dt_logits.')
    if dt_logits is not None and alpha_log_rate is None:
        raise ValueError('dt_logits requires alpha_log_rate.')
    if log_alpha is not None and (alpha_log_rate is not None or dt_bias is not None):
        raise ValueError('alpha_log_rate and dt_bias require dt_logits.')
    decay_input = dt_logits if dt_logits is not None else log_alpha
    if q.ndim != 4 or q.shape != k.shape or q.shape != decay_input.shape:
        raise ValueError('q, k, and the decay input must have the same [B, T, H, K] shape.')
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('v must have shape [B, T, H, V] with the same B, T, and H as q.')
    if min(q.shape) < 1 or not 1 <= q.shape[-1] <= 128 or not 1 <= v.shape[-1] <= 128:
        raise ValueError('DDN requires nonempty inputs and key/value head dimensions in [1, 128].')
    for name, x in (('q', q), ('k', k), ('v', v)):
        if x.dtype != torch.bfloat16:
            raise ValueError(f'{name} must use bfloat16; float16 and float32 q/k/v inputs are unsupported.')
    if q.device.type != 'cuda':
        raise ValueError('DDN Triton kernels require CUDA inputs.')
    if torch.version.hip is not None:
        raise ValueError('DDN currently supports the NVIDIA Triton backend.')
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError('DDN requires an NVIDIA GPU with compute capability 8.0 or newer.')
    tensors = (k, v, decay_input, beta_logits, gamma_logits, initial_state)
    for x in tensors:
        if x is not None and (x.device != q.device or x.dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError('All inputs must share a CUDA device and use float32 or bfloat16.')
    for name, x in (('beta_logits', beta_logits), ('gamma_logits', gamma_logits)):
        if x is None or x.shape not in (q.shape[:-1], (*q.shape[:-1], 1)):
            raise ValueError(f'{name} must have shape [B, T, H] or [B, T, H, 1].')
    if chunk_size != 32:
        raise ValueError('DDN currently requires chunk_size=32.')
    if not math.isfinite(transition_eps) or transition_eps <= 0:
        raise ValueError('transition_eps must be finite and positive.')
    if scale is not None and not math.isfinite(scale):
        raise ValueError('scale must be finite.')
    sequences = q.shape[0]
    if cu_seqlens is not None:
        if (q.shape[0] != 1 or cu_seqlens.ndim != 1 or len(cu_seqlens) < 2
                or cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.device != q.device):
            raise ValueError('Packed inputs require B=1 and a CUDA int32/int64 cu_seqlens vector.')
        sequences = len(cu_seqlens) - 1
        if cu_seqlens_cpu is not None:
            if cu_seqlens_cpu.device.type != 'cpu' or cu_seqlens_cpu.shape != cu_seqlens.shape:
                raise ValueError('cu_seqlens_cpu must be a CPU copy of cu_seqlens.')
            offsets = cu_seqlens_cpu.tolist()
            if offsets[0] != 0 or offsets[-1] != q.shape[1] or any(start > end for start, end in zip(offsets, offsets[1:])):
                raise ValueError('Sequence offsets must start at zero, be nondecreasing, and end at T.')
    elif cu_seqlens_cpu is not None:
        raise ValueError('cu_seqlens_cpu requires cu_seqlens.')
    if initial_state is not None and initial_state.shape != (sequences, q.shape[2], q.shape[-1], v.shape[-1]):
        raise ValueError('initial_state must have shape [N, H, K, V], where N is the number of sequences.')
    if direction_logits is None:
        raise ValueError('DDN requires direction_logits.')
    if (direction_logits.shape != v.shape or direction_logits.device != v.device
            or direction_logits.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError('direction_logits must match v shape/device and use bfloat16 or float32.')

    if alpha_log_rate is not None:
        if alpha_log_rate.shape != (q.shape[2],):
            raise ValueError('alpha_log_rate must have shape [H].')
        if dt_bias is not None and dt_bias.numel() != q.shape[2] * q.shape[3]:
            raise ValueError('dt_bias must have H * K elements.')
        for name, tensor in (('alpha_log_rate', alpha_log_rate), ('dt_bias', dt_bias)):
            if tensor is not None and (tensor.device != q.device or tensor.dtype not in (torch.float32, torch.bfloat16)):
                raise ValueError(f'{name} must share the input device and use float32 or bfloat16.')
    elif dt_bias is not None:
        raise ValueError('dt_bias requires alpha_log_rate.')


@torch.compiler.disable
@input_guard(no_guard_contiguous=("beta_logits", "gamma_logits"))
def chunk_ddn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   beta_logits: torch.Tensor, gamma_logits: torch.Tensor,
                   log_alpha: torch.Tensor | None = None,
                   scale: float | None = None, initial_state: torch.Tensor | None = None,
                   output_final_state: bool = False, use_qk_l2norm_in_kernel: bool = False,
                   cu_seqlens: torch.Tensor | None = None, cu_seqlens_cpu: torch.Tensor | None = None,
                   transition_eps: float = 1e-6, chunk_size: int = 32, direction_logits: torch.Tensor | None = None,
                   disable_recompute: bool = False, alpha_log_rate: torch.Tensor | None = None,
                   dt_bias: torch.Tensor | None = None, dt_logits: torch.Tensor | None = None):
    validate_inputs(q, k, v, log_alpha, beta_logits, gamma_logits, initial_state,
                    cu_seqlens, cu_seqlens_cpu, chunk_size, transition_eps, scale,
                    direction_logits, alpha_log_rate=alpha_log_rate, dt_bias=dt_bias, dt_logits=dt_logits)
    dtype = q.dtype
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    layout = make_layout(q, cu_seqlens, cu_seqlens_cpu)
    return ChunkDDNFunction.apply(
        q, k, v, dt_logits if dt_logits is not None else log_alpha, beta_logits, gamma_logits, direction_logits,
        initial_state, output_final_state, layout, dtype,
        use_qk_l2norm_in_kernel, transition_eps, scale, disable_recompute, alpha_log_rate, dt_bias,
    )
