# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


@tilelang.jit
def _build_boundary_state_fwd_kernel(
    num_sequences: int,
    num_chunks: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    max_chunks: int,
    value_tile: int,
    threads: int,
    packed: bool,
    use_initial_state: bool,
    store_final_state: bool,
    save_psi: bool,
):
    dtype = T.bfloat16
    chunks_per_sequence = num_chunks // num_sequences if not packed else max_chunks
    workspace_cols = max(key_dim, value_dim)
    initial_shape = (
        (num_sequences, num_heads, key_dim, value_dim)
        if use_initial_state
        else (1,)
    )
    final_shape = (
        (num_sequences, num_heads, key_dim, value_dim)
        if store_final_state
        else (1,)
    )
    psi_shape = (
        (num_chunks, num_heads, 32, value_dim)
        if save_psi
        else (1,)
    )

    @T.prim_func
    def kernel(
        bar_U: T.Tensor((num_chunks, num_heads, 32, key_dim), dtype),
        N_Z: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        F_R: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        hat_V: T.Tensor((num_chunks, num_heads, 32, value_dim), dtype),
        Pi_C_H: T.Tensor((num_chunks, num_heads, key_dim, 32), dtype),
        log_pi_C: T.Tensor((num_chunks, num_heads, key_dim), T.float32),
        S_0: T.Tensor(initial_shape, T.float32),
        S_n: T.Tensor((num_chunks, num_heads, key_dim, value_dim), dtype),
        S_N: T.Tensor(final_shape, T.float32),
        Psi: T.Tensor(psi_shape, dtype),
        chunk_offsets: T.Tensor((num_sequences + 1,), T.int64),
    ):
        with T.Kernel(num_sequences, num_heads, threads=threads) as (seq_idx, head_idx):
            T.annotate_min_blocks_per_sm(1)
            S_shared = T.alloc_shared((key_dim, value_dim), dtype)
            key_token_workspace_shared = T.alloc_shared((key_dim, 32), dtype)
            # reuse the token workspace for F_R, hat_V, bar_U, and Psi.
            token_workspace_shared = T.alloc_shared(
                (32, workspace_cols), dtype
            )
            pi_C_shared = T.alloc_shared((key_dim,), T.float32)

            S_F_R_T_fragment = T.alloc_fragment((key_dim, 32), T.float32)
            R_fragment = T.alloc_fragment((key_dim, value_tile), T.float32)
            Psi_fragment = T.alloc_fragment((32, value_tile), T.float32)
            Pi_C_H_Psi_fragment = T.alloc_fragment(
                (key_dim, value_tile), T.float32
            )

            if use_initial_state:
                T.copy(S_0[seq_idx, head_idx, :, :], S_shared)
            else:
                T.clear(S_shared)

            if packed:
                sequence_start = chunk_offsets[seq_idx]
                sequence_end = chunk_offsets[seq_idx + 1]
                sequence_chunks = sequence_end - sequence_start
            else:
                sequence_start = seq_idx * chunks_per_sequence

            for chunk_step in T.serial(max_chunks):
                if packed:
                    active = chunk_step < sequence_chunks
                    chunk_idx = T.if_then_else(active, sequence_start + chunk_step, 0)
                else:
                    active = True
                    chunk_idx = sequence_start + chunk_step

                if active:
                    for key_idx, value_idx in T.Parallel(key_dim, value_dim):
                        S_n[chunk_idx, head_idx, key_idx, value_idx] = S_shared[key_idx, value_idx]

                    # Once S_[n] F_R^T is complete, S_[n] can be overwritten by R.
                    T.copy(
                        F_R[chunk_idx, head_idx, :, :],
                        token_workspace_shared[:, :value_dim],
                    )
                    T.gemm(
                        S_shared,
                        token_workspace_shared[:, :value_dim],
                        S_F_R_T_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.copy(S_F_R_T_fragment, key_token_workspace_shared)

                    for value_tile_idx in T.serial(T.ceildiv(value_dim, value_tile)):
                        value_start = value_tile_idx * value_tile
                        T.copy(
                            hat_V[chunk_idx, head_idx, :, value_start: value_start + value_tile],
                            token_workspace_shared[:, :value_tile],
                        )
                        T.gemm(
                            key_token_workspace_shared,
                            token_workspace_shared[:, :value_tile],
                            R_fragment,
                            clear_accum=True,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, value_tile):
                            R_fragment[key_idx, value_idx] = (
                                T.cast(
                                    S_shared[key_idx, value_start + value_idx],
                                    T.float32,
                                )
                                - R_fragment[key_idx, value_idx]
                            )
                            S_shared[key_idx, value_start + value_idx] = T.cast(
                                R_fragment[key_idx, value_idx], dtype
                            )

                    for key_idx in T.Parallel(key_dim):
                        pi_C_shared[key_idx] = T.exp(log_pi_C[chunk_idx, head_idx, key_idx])
                    T.copy(
                        Pi_C_H[chunk_idx, head_idx, :, :],
                        key_token_workspace_shared,
                    )

                    # Psi = N_Z - bar_U R; S_[n+1] = pi^C R + (pi^C H) Psi.
                    for value_tile_idx in T.serial(T.ceildiv(value_dim, value_tile)):
                        value_start = value_tile_idx * value_tile
                        T.copy(
                            bar_U[chunk_idx, head_idx, :, :],
                            token_workspace_shared[:, :key_dim],
                        )
                        T.gemm(
                            token_workspace_shared[:, :key_dim],
                            S_shared[:, value_start: value_start + value_tile],
                            Psi_fragment,
                            clear_accum=True,
                        )
                        for token_idx, value_idx in T.Parallel(32, value_tile):
                            Psi_fragment[token_idx, value_idx] = (
                                T.cast(
                                    N_Z[chunk_idx, head_idx, token_idx, value_start + value_idx],
                                    T.float32,
                                )
                                - Psi_fragment[token_idx, value_idx]
                            )
                            if save_psi:
                                Psi[chunk_idx, head_idx, token_idx, value_start + value_idx] = T.cast(
                                    Psi_fragment[token_idx, value_idx], dtype
                                )
                        T.copy(
                            Psi_fragment,
                            token_workspace_shared[:, :value_tile],
                        )
                        T.gemm(
                            key_token_workspace_shared,
                            token_workspace_shared[:, :value_tile],
                            Pi_C_H_Psi_fragment,
                            clear_accum=True,
                        )
                        for key_idx, value_idx in T.Parallel(key_dim, value_tile):
                            S_n_plus_1 = (
                                pi_C_shared[key_idx]
                                * T.cast(
                                    S_shared[key_idx, value_start + value_idx],
                                    T.float32,
                                )
                                + Pi_C_H_Psi_fragment[key_idx, value_idx]
                            )
                            S_shared[key_idx, value_start + value_idx] = T.cast(
                                S_n_plus_1, dtype
                            )

            if store_final_state:
                for key_idx, value_idx in T.Parallel(key_dim, value_dim):
                    S_N[seq_idx, head_idx, key_idx, value_idx] = T.cast(
                        S_shared[key_idx, value_idx], T.float32
                    )

    return kernel


def boundary_state_fwd(
    bar_U,
    N_Z,
    F_R,
    hat_V,
    Pi_C_H,
    log_pi_C,
    S_0,
    output_final_state,
    layout,
    save_psi,
):
    dk, dv = bar_U.shape[-1], hat_V.shape[-1]
    if dk > 128 or dv > 128:
        raise ValueError("The TileLang kernel requires DK and DV at most 128.")

    output_dtype = hat_V.dtype
    bar_U_bf16 = bar_U.to(torch.bfloat16)
    N_Z_bf16 = N_Z.to(torch.bfloat16)
    F_R_bf16 = F_R.to(torch.bfloat16)
    hat_V_bf16 = hat_V.to(torch.bfloat16)
    Pi_C_H_bf16 = Pi_C_H.to(torch.bfloat16)

    padded_dk = 1 << max(4, (dk - 1).bit_length())
    padded_dv = (dv + 31) // 32 * 32
    pad_k, pad_v = padded_dk - dk, padded_dv - dv
    if pad_k:
        bar_U_bf16 = F.pad(bar_U_bf16, (0, pad_k))
        Pi_C_H_bf16 = F.pad(Pi_C_H_bf16, (0, 0, 0, pad_k))
        log_pi_C = F.pad(log_pi_C, (0, pad_k))
    if pad_v:
        N_Z_bf16 = F.pad(N_Z_bf16, (0, pad_v))
        F_R_bf16 = F.pad(F_R_bf16, (0, pad_v))
        hat_V_bf16 = F.pad(hat_V_bf16, (0, pad_v))
    if S_0 is not None:
        S_0 = S_0.float()
        if pad_k or pad_v:
            S_0 = F.pad(S_0, (0, pad_v, 0, pad_k))

    S_n = torch.empty(
        layout.chunks,
        layout.heads,
        padded_dk,
        padded_dv,
        device=bar_U.device,
        dtype=torch.bfloat16,
    )
    S_N = (
        torch.empty(
            layout.sequences,
            layout.heads,
            padded_dk,
            padded_dv,
            device=bar_U.device,
            dtype=torch.float32,
        )
        if output_final_state
        else None
    )
    Psi = torch.empty_like(N_Z_bf16) if save_psi else None
    packed = layout.cu_seqlens is not None
    chunk_offsets = (
        layout.offsets.to(dtype=torch.int64)
        if packed
        else torch.empty(layout.sequences + 1, device=bar_U.device, dtype=torch.int64)
    )

    S_0_arg = S_0 if S_0 is not None else torch.empty(1, device=bar_U.device)
    S_N_arg = S_N if S_N is not None else torch.empty(1, device=bar_U.device)
    Psi_arg = (
        Psi
        if Psi is not None
        else torch.empty(1, device=bar_U.device, dtype=torch.bfloat16)
    )
    kernel = _build_boundary_state_fwd_kernel(
        layout.sequences,
        layout.chunks,
        layout.heads,
        padded_dk,
        padded_dv,
        layout.max_chunks,
        32,
        32 if padded_dk == 16 else 256,
        packed,
        S_0 is not None,
        output_final_state,
        save_psi,
    )
    kernel(
        bar_U_bf16,
        N_Z_bf16,
        F_R_bf16,
        hat_V_bf16,
        Pi_C_H_bf16,
        log_pi_C,
        S_0_arg,
        S_n,
        S_N_arg,
        Psi_arg,
        chunk_offsets,
    )

    if pad_k or pad_v:
        S_n = S_n[..., :dk, :dv].contiguous()
        if S_N is not None:
            S_N = S_N[..., :dk, :dv].contiguous()
        if Psi is not None:
            Psi = Psi[..., :dv].contiguous()
    if Psi is not None:
        Psi = Psi.to(output_dtype)
    return S_n, S_N, Psi
