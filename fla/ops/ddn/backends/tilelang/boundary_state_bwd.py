# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True})
def _build_boundary_state_bwd_kernel(
    num_sequences: int,
    num_chunks: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    max_chunks: int,
    threads: int,
    packed: bool,
    has_dS_N: bool,
    store_dS_0: bool,
):
    dtype = T.bfloat16
    chunks_per_sequence = num_chunks // num_sequences if not packed else max_chunks
    dS_N_shape = (
        (num_sequences, num_heads, key_dim, value_dim) if has_dS_N else (1,)
    )
    dS_0_shape = (
        (num_sequences, num_heads, key_dim, value_dim) if store_dS_0 else (1,)
    )

    @T.prim_func
    def kernel(
        S_n: T.Tensor((num_chunks, num_heads, key_dim, value_dim), dtype),
        Psi: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        Pi_C_H: T.Tensor((num_chunks, num_heads, key_dim, 32), dtype),
        bar_U: T.Tensor((num_chunks, num_heads, 32, key_dim), dtype),
        F_R: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        hat_V: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        log_pi_C: T.Tensor((num_chunks, num_heads, key_dim), T.float32),
        dS_n_readout: T.Tensor((num_chunks, num_heads, key_dim, value_dim), dtype),
        dS_N: T.Tensor(dS_N_shape, T.float32),
        dF_R_in: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        dhat_V_in: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        dS_0: T.Tensor(dS_0_shape, T.float32),
        dPi_C_H: T.Tensor((num_chunks, num_heads, key_dim, 32), dtype),
        dN_Z: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        dbar_U: T.Tensor((num_chunks, num_heads, 32, key_dim), dtype),
        dF_R: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        dhat_V: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        dlog_pi_C: T.Tensor((num_chunks, num_heads, key_dim), T.float32),
        chunk_offsets: T.Tensor((num_sequences + 1,), T.int64),
    ):
        with T.Kernel(num_sequences, num_heads, threads=threads) as (seq_idx, head_idx):
            T.annotate_min_blocks_per_sm(1)
            dS_shared = T.alloc_shared((key_dim, value_dim), T.float32)
            key_token_workspace_shared = T.alloc_shared((key_dim, 32), dtype)
            dR_hat_V_T_shared = T.alloc_shared((key_dim, 32), dtype)
            key_value_shared = T.alloc_shared((key_dim, 32), dtype)
            bar_U_shared = T.alloc_shared((32, key_dim), dtype)
            token_value_shared = T.alloc_shared((32, 32), dtype)

            dPi_C_H_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            dPsi_fragment = T.alloc_fragment((32, 32), T.float32)
            S_n_F_R_T_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            R_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            dR_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            dbar_U_fragment = T.alloc_fragment((32, key_dim), T.float32)
            dR_hat_V_T_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            matrix_fragment = T.alloc_fragment((32, 32), T.float32)
            dlog_product_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            dlog_tile_fragment = T.alloc_fragment((key_dim,), T.float32)
            dlog_pi_C_fragment = T.alloc_fragment((key_dim,), T.float32)
            pi_C_fragment = T.alloc_fragment((key_dim,), T.float32)

            if has_dS_N:
                T.copy(dS_N[seq_idx, head_idx, :, :], dS_shared)
            else:
                T.clear(dS_shared)

            if packed:
                sequence_start = chunk_offsets[seq_idx]
                sequence_end = chunk_offsets[seq_idx + 1]
                sequence_chunks = sequence_end - sequence_start
            else:
                sequence_start = seq_idx * chunks_per_sequence

            for chunk_step in T.serial(max_chunks):
                if packed:
                    active = chunk_step < sequence_chunks
                    chunk_idx = T.if_then_else(active, sequence_end - 1 - chunk_step, 0)
                else:
                    active = True
                    chunk_idx = sequence_start + max_chunks - 1 - chunk_step

                if active:
                    T.clear(dPi_C_H_fragment)
                    # stage Pi here so the reduction workspace can reuse dR_hat_V_T_shared
                    T.copy(Pi_C_H[chunk_idx, head_idx, :, :], key_token_workspace_shared)
                    for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                        value_start = value_tile_idx * 32
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            key_value_shared[key_idx, value_idx] = T.cast(
                                dS_shared[key_idx, value_start + value_idx], dtype
                            )
                        T.copy(
                            Psi[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        T.gemm(
                            key_value_shared,
                            token_value_shared,
                            dPi_C_H_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            key_token_workspace_shared,
                            key_value_shared,
                            dPsi_fragment,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        T.copy(
                            dPsi_fragment,
                            dN_Z[chunk_idx, head_idx, :, value_start: value_start + 32],
                        )
                    T.copy(dPi_C_H_fragment, dPi_C_H[chunk_idx, head_idx, :, :])

                    T.clear(S_n_F_R_T_fragment)
                    for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                        value_start = value_tile_idx * 32
                        T.copy(
                            S_n[chunk_idx, head_idx, :, value_start: value_start + 32],
                            key_value_shared,
                        )
                        T.copy(
                            F_R[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        T.gemm(
                            key_value_shared,
                            token_value_shared,
                            S_n_F_R_T_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    T.copy(S_n_F_R_T_fragment, key_token_workspace_shared)
                    T.copy(bar_U[chunk_idx, head_idx, :, :], bar_U_shared)
                    for key_idx in T.Parallel(key_dim):
                        pi_C_fragment[key_idx] = T.exp(log_pi_C[chunk_idx, head_idx, key_idx])

                    T.clear(dbar_U_fragment)
                    T.clear(dlog_pi_C_fragment)
                    for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                        value_start = value_tile_idx * 32
                        T.copy(
                            S_n[chunk_idx, head_idx, :, value_start: value_start + 32],
                            key_value_shared,
                        )
                        T.copy(
                            hat_V[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        T.gemm(
                            key_token_workspace_shared,
                            token_value_shared,
                            R_fragment,
                            clear_accum=True,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            R_fragment[key_idx, value_idx] = (
                                T.cast(key_value_shared[key_idx, value_idx], T.float32)
                                - R_fragment[key_idx, value_idx]
                            )
                        T.copy(
                            dN_Z[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        T.gemm(
                            bar_U_shared,
                            token_value_shared,
                            dR_fragment,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            dR_fragment[key_idx, value_idx] = (
                                pi_C_fragment[key_idx]
                                * dS_shared[key_idx, value_start + value_idx]
                                - dR_fragment[key_idx, value_idx]
                            )
                            dlog_product_fragment[key_idx, value_idx] = (
                                dS_shared[key_idx, value_start + value_idx]
                                * R_fragment[key_idx, value_idx]
                            )
                            # The current state gradient has no further uses in this tile.
                            dS_shared[key_idx, value_start + value_idx] = dR_fragment[key_idx, value_idx]
                        T.reduce_sum(
                            dlog_product_fragment,
                            dlog_tile_fragment,
                            dim=1,
                            clear=True,
                        )
                        for key_idx in T.Parallel(key_dim):
                            dlog_pi_C_fragment[key_idx] += dlog_tile_fragment[key_idx]
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            key_value_shared[key_idx, value_idx] = T.cast(
                                R_fragment[key_idx, value_idx], dtype
                            )
                        T.gemm(
                            token_value_shared,
                            key_value_shared,
                            dbar_U_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    for token_idx, key_idx in T.Parallel(32, key_dim):
                        dbar_U[chunk_idx, head_idx, token_idx, key_idx] = T.cast(
                            -dbar_U_fragment[token_idx, key_idx], dtype
                        )
                    for key_idx in T.Parallel(key_dim):
                        dlog_pi_C[chunk_idx, head_idx, key_idx] = (
                            pi_C_fragment[key_idx]
                            * dlog_pi_C_fragment[key_idx]
                        )

                    T.clear(dR_hat_V_T_fragment)
                    for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                        value_start = value_tile_idx * 32
                        T.copy(
                            hat_V[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            key_value_shared[key_idx, value_idx] = T.cast(
                                dS_shared[key_idx, value_start + value_idx], dtype
                            )
                        T.gemm(
                            key_value_shared,
                            token_value_shared,
                            dR_hat_V_T_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    T.copy(dR_hat_V_T_fragment, dR_hat_V_T_shared)

                    for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                        value_start = value_tile_idx * 32
                        T.copy(
                            S_n[chunk_idx, head_idx, :, value_start: value_start + 32],
                            key_value_shared,
                        )
                        T.copy(
                            dS_shared[:, value_start: value_start + 32],
                            dR_fragment,
                        )

                        T.gemm(
                            dR_hat_V_T_shared,
                            key_value_shared,
                            matrix_fragment,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        for token_idx, value_idx in T.Parallel(32, 32):
                            dF_R[chunk_idx, head_idx, token_idx, value_start + value_idx] = T.cast(
                                T.cast(
                                    dF_R_in[chunk_idx, head_idx, token_idx, value_start + value_idx],
                                    T.float32,
                                )
                                - matrix_fragment[token_idx, value_idx],
                                dtype,
                            )
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            key_value_shared[key_idx, value_idx] = T.cast(
                                dR_fragment[key_idx, value_idx], dtype
                            )
                        T.gemm(
                            key_token_workspace_shared,
                            key_value_shared,
                            matrix_fragment,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        for token_idx, value_idx in T.Parallel(32, 32):
                            dhat_V[chunk_idx, head_idx, token_idx, value_start + value_idx] = T.cast(
                                T.cast(
                                    dhat_V_in[chunk_idx, head_idx, token_idx, value_start + value_idx],
                                    T.float32,
                                )
                                - matrix_fragment[token_idx, value_idx],
                                dtype,
                            )
                        T.copy(
                            F_R[chunk_idx, head_idx, :, value_start: value_start + 32],
                            token_value_shared,
                        )
                        T.gemm(
                            dR_hat_V_T_shared,
                            token_value_shared,
                            R_fragment,
                            clear_accum=True,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, 32):
                            dS_shared[key_idx, value_start + value_idx] = (
                                dR_fragment[key_idx, value_idx]
                                - R_fragment[key_idx, value_idx]
                                + T.cast(
                                    dS_n_readout[
                                        chunk_idx, head_idx, key_idx, value_start + value_idx
                                    ],
                                    T.float32,
                                )
                            )

            if store_dS_0:
                T.copy(dS_shared, dS_0[seq_idx, head_idx, :, :])

    return kernel


def boundary_state_bwd(
    S_n, Psi, Pi_C_H, bar_U, F_R, hat_V, log_pi_C,
    dS_n_readout, dS_N, dF_R, dhat_V, S_0, layout,
):
    dk, dv = S_n.shape[-2:]
    if dk > 128 or dv > 128:
        raise ValueError("The TileLang boundary-state backward kernel requires DK and DV at most 128.")

    padded_dk = 1 << max(4, (dk - 1).bit_length())
    padded_dv = (dv + 31) // 32 * 32
    pad_k, pad_v = padded_dk - dk, padded_dv - dv

    def pad_state(x):
        return F.pad(x, (0, pad_v, 0, pad_k)) if pad_k or pad_v else x

    def pad_key(x):
        return F.pad(x, (0, pad_k)) if pad_k else x

    def pad_value(x):
        return F.pad(x, (0, pad_v)) if pad_v else x

    S_n = pad_state(S_n.to(torch.bfloat16))
    Psi = pad_value(Psi.to(torch.bfloat16))
    Pi_C_H = F.pad(Pi_C_H.to(torch.bfloat16), (0, 0, 0, pad_k)) if pad_k else Pi_C_H.to(torch.bfloat16)
    bar_U = pad_key(bar_U.to(torch.bfloat16))
    F_R = pad_value(F_R.to(torch.bfloat16))
    hat_V = pad_value(hat_V.to(torch.bfloat16))
    log_pi_C = pad_key(log_pi_C)
    dS_n_readout = pad_state(dS_n_readout.to(torch.bfloat16))
    dF_R = pad_value(dF_R.to(torch.bfloat16))
    dhat_V = pad_value(dhat_V.to(torch.bfloat16))
    if dS_N is not None:
        dS_N = pad_state(dS_N.float())

    dS_0 = (
        torch.empty(
            layout.sequences, layout.heads, padded_dk, padded_dv,
            device=S_n.device, dtype=torch.float32,
        )
        if S_0 is not None else None
    )
    dPi_C_H = torch.empty_like(Pi_C_H)
    dN_Z = torch.empty_like(Psi)
    dbar_U = torch.empty_like(bar_U)
    dF_R_out = torch.empty_like(F_R)
    dhat_V_out = torch.empty_like(hat_V)
    dlog_pi_C = torch.empty_like(log_pi_C)
    packed = layout.cu_seqlens is not None
    chunk_offsets = (
        layout.offsets.to(dtype=torch.int64)
        if packed else torch.empty(layout.sequences + 1, device=S_n.device, dtype=torch.int64)
    )
    dS_N_arg = dS_N if dS_N is not None else torch.empty(1, device=S_n.device)
    dS_0_arg = dS_0 if dS_0 is not None else torch.empty(1, device=S_n.device)
    kernel = _build_boundary_state_bwd_kernel(
        layout.sequences,
        layout.chunks,
        layout.heads,
        padded_dk,
        padded_dv,
        layout.max_chunks,
        32 if padded_dk == 16 else 256,
        packed,
        dS_N is not None,
        S_0 is not None,
    )
    kernel(
        S_n, Psi, Pi_C_H, bar_U, F_R, hat_V, log_pi_C,
        dS_n_readout, dS_N_arg, dF_R, dhat_V,
        dS_0_arg, dPi_C_H, dN_Z, dbar_U, dF_R_out, dhat_V_out, dlog_pi_C,
        chunk_offsets,
    )

    if pad_k or pad_v:
        if dS_0 is not None:
            dS_0 = dS_0[..., :dk, :dv].contiguous()
        dPi_C_H = dPi_C_H[..., :dk, :].contiguous()
        dN_Z = dN_Z[..., :dv].contiguous()
        dbar_U = dbar_U[..., :dk].contiguous()
        dF_R_out = dF_R_out[..., :dv].contiguous()
        dhat_V_out = dhat_V_out[..., :dv].contiguous()
        dlog_pi_C = dlog_pi_C[..., :dk].contiguous()
    return dS_0, dPi_C_H, dN_Z, dbar_U, dF_R_out, dhat_V_out, dlog_pi_C
