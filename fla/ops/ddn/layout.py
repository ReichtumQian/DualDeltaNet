# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import get_max_num_splits, prepare_chunk_indices, prepare_chunk_offsets


@dataclass(frozen=True)
class ChunkLayout:
    batch: int
    length: int
    heads: int
    chunks: int
    sequences: int
    max_chunks: int
    cu_seqlens: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    offsets: torch.Tensor | None = None

    @property
    def shape(self):
        return (self.chunks, self.heads, 32)


def make_layout(q, cu_seqlens=None, cu_seqlens_cpu=None):
    b, t, h, _ = q.shape
    if cu_seqlens is None:
        return ChunkLayout(b, t, h, b * triton.cdiv(t, 32), b, triton.cdiv(t, 32))
    indices = prepare_chunk_indices(cu_seqlens, 32, cu_seqlens_cpu)
    offsets = prepare_chunk_offsets(cu_seqlens, 32)
    return ChunkLayout(b, t, h, len(indices), len(cu_seqlens) - 1,
                       get_max_num_splits(cu_seqlens, 32, cu_seqlens_cpu), cu_seqlens, indices, offsets)


@triton.jit
def chunk_bounds(i, Cu, Indices, H: tl.constexpr, T: tl.constexpr, PACKED: tl.constexpr):
    chunk = i // H
    head = i % H
    if PACKED:
        seq = tl.load(Indices + chunk * 2).to(tl.int64)
        local = tl.load(Indices + chunk * 2 + 1).to(tl.int64)
        bos = tl.load(Cu + seq).to(tl.int64) + local * 32
        eos = tl.load(Cu + seq + 1).to(tl.int64)
    else:
        seq = chunk // tl.cdiv(T, 32)
        bos = seq * T + (chunk % tl.cdiv(T, 32)) * 32
        eos = (seq + 1) * T
    return head, bos, eos


@triton.jit
def sequence_chunks(seq, Offsets, T: tl.constexpr, PACKED: tl.constexpr):
    if PACKED:
        start = tl.load(Offsets + seq).to(tl.int64)
        end = tl.load(Offsets + seq + 1).to(tl.int64)
    else:
        start = seq * tl.cdiv(T, 32)
        end = start + tl.cdiv(T, 32)
    return start, end
