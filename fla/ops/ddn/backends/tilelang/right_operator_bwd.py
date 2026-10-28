# Copyright (c) 2026, Yixiao Qian, Yaxin Zhang

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _build_right_operator_bwd_kernel(num_chunks: int, value_dim: int):
    value_tile = 32

    dtype = T.bfloat16

    @T.prim_func
    def kernel(
        M_R: T.Tensor((num_chunks, 32, 32), dtype),
        G_L: T.Tensor((num_chunks, 32, 32), dtype),
        eta: T.Tensor((num_chunks, 32), T.float32),
        gamma: T.Tensor((num_chunks, 32), T.float32),
        E: T.Tensor((num_chunks, 32, value_dim), dtype),
        hat_V: T.Tensor((num_chunks, 32, value_dim), dtype),
        dPhi: T.Tensor((num_chunks, 32, 32), dtype),
        dF_R: T.Tensor((num_chunks, 32, value_dim), dtype),
        dN_Z: T.Tensor((num_chunks, 32, value_dim), dtype),
        dhat_V_in: T.Tensor((num_chunks, 32, value_dim), dtype),
        dG_L: T.Tensor((num_chunks, 32, 32), dtype),
        deta: T.Tensor((num_chunks, 32), T.float32),
        dgamma: T.Tensor((num_chunks, 32), T.float32),
        dE: T.Tensor((num_chunks, 32, value_dim), dtype),
        dhat_V: T.Tensor((num_chunks, 32, value_dim), dtype),
    ):
        with T.Kernel(num_chunks, threads=128) as block_idx:
            chunk_idx = T.cast(block_idx, T.int64)
            dgamma_E_shared = T.alloc_shared((32, T.ceildiv(value_dim, 32) * 32), dtype)
            dhat_V_shared = T.alloc_shared((32, T.ceildiv(value_dim, 32) * 32), dtype)
            M_R_shared = T.alloc_shared((32, 32), dtype)
            matrix_a_shared = T.alloc_shared((32, 32), dtype)
            matrix_b_shared = T.alloc_shared((32, 32), dtype)
            hat_V_shared = T.alloc_shared((32, 32), dtype)
            dF_R_shared = T.alloc_shared((32, 32), dtype)
            gamma_E_shared = T.alloc_shared((32, 32), dtype)
            M_R_fp32_shared = T.alloc_shared((32, 32), T.float32)
            dM_fp32_shared = T.alloc_shared((32, 32), T.float32)
            matrix_fp32_workspace_shared = T.alloc_shared((32, 32), T.float32)
            dT_shared = T.alloc_shared((32, 32), T.float32)
            matrix_fragment = T.alloc_fragment((32, 32), T.float32)
            dM_fragment = T.alloc_fragment((32, 32), T.float32)
            product_fragment = T.alloc_fragment((32, 32), T.float32)
            reduction_fragment = T.alloc_fragment((32,), T.float32)
            deta_acc_shared = T.alloc_shared((32,), T.float32)
            deta_initial_fragment = T.alloc_fragment((32,), T.float32)
            dgamma_tile_fragment = T.alloc_fragment((32,), T.float32)
            dgamma_acc_fragment = T.alloc_fragment((32,), T.float32)
            eta_shared = T.alloc_shared((32,), T.float32)
            gamma_shared = T.alloc_shared((32,), T.float32)
            value_workspace_shared = T.alloc_shared((32, value_tile), T.float32)
            value_gradient_fragment = T.alloc_fragment((32, value_tile), T.float32)
            dgamma_product_fragment = T.alloc_fragment((32, value_tile), T.float32)

            T.copy(M_R[chunk_idx, :, :], M_R_shared)
            T.copy(dPhi[chunk_idx, :, :], matrix_a_shared)
            T.copy(eta[chunk_idx, :], eta_shared)
            T.copy(gamma[chunk_idx, :], gamma_shared)
            T.gemm(matrix_a_shared, M_R_shared, matrix_fragment, clear_accum=True)
            for row_idx, col_idx in T.Parallel(32, 32):
                dG_L[chunk_idx, row_idx, col_idx] = matrix_fragment[row_idx, col_idx] * eta_shared[col_idx]
                product_fragment[row_idx, col_idx] = (
                    matrix_fragment[row_idx, col_idx] * T.cast(G_L[chunk_idx, row_idx, col_idx], T.float32)
                )
                matrix_b_shared[row_idx, col_idx] = T.cast(G_L[chunk_idx, row_idx, col_idx], T.float32) * eta_shared[col_idx]
            T.reduce_sum(product_fragment, deta_initial_fragment, dim=0)
            T.copy(deta_initial_fragment, deta_acc_shared)
            T.gemm(matrix_a_shared, matrix_b_shared, dM_fragment, transpose_A=True, clear_accum=True)

            for value_tile_idx in T.serial(T.ceildiv(value_dim, 32)):
                T.copy(hat_V[chunk_idx, :, value_tile_idx * 32:value_tile_idx * 32 + 32], hat_V_shared)
                T.gemm(M_R_shared, hat_V_shared, matrix_fragment, transpose_A=True, clear_accum=True)
                for row_idx, col_idx in T.Parallel(32, 32):
                    nz = T.if_then_else(
                        value_tile_idx * 32 + col_idx < value_dim,
                        T.cast(dN_Z[chunk_idx, row_idx, value_tile_idx * 32 + col_idx], T.float32),
                        0.0,
                    )
                    product_fragment[row_idx, col_idx] = nz * matrix_fragment[row_idx, col_idx]
                    matrix_b_shared[row_idx, col_idx] = eta_shared[row_idx] * nz
                T.reduce_sum(product_fragment, reduction_fragment, dim=1)
                for row_idx in T.Parallel(32):
                    deta_acc_shared[row_idx] += reduction_fragment[row_idx]
                T.gemm(hat_V_shared, matrix_b_shared, dM_fragment, transpose_B=True, clear_accum=False)
                T.copy(dhat_V_in[chunk_idx, :, value_tile_idx * 32:value_tile_idx * 32 + 32], matrix_fragment)
                T.gemm(M_R_shared, matrix_b_shared, matrix_fragment, clear_accum=False)
                T.copy(matrix_fragment, dhat_V_shared[:, value_tile_idx * 32:value_tile_idx * 32 + 32])

                T.copy(dF_R[chunk_idx, :, value_tile_idx * 32:value_tile_idx * 32 + 32], dF_R_shared)
                for row_idx, col_idx in T.Parallel(32, 32):
                    e = T.if_then_else(
                        value_tile_idx * 32 + col_idx < value_dim,
                        T.cast(E[chunk_idx, row_idx, value_tile_idx * 32 + col_idx], T.float32),
                        0.0,
                    )
                    gamma_E_shared[row_idx, col_idx] = gamma_shared[row_idx] * e
                T.gemm(dF_R_shared, gamma_E_shared, dM_fragment, transpose_B=True, clear_accum=False)
                T.copy(dF_R_shared, matrix_a_shared)
                T.gemm(M_R_shared, matrix_a_shared, matrix_fragment, transpose_A=True, clear_accum=True)
                T.copy(matrix_fragment, dgamma_E_shared[:, value_tile_idx * 32:value_tile_idx * 32 + 32])
            T.copy(deta_acc_shared, deta[chunk_idx, :])

            for row_idx, col_idx in T.Parallel(32, 32):
                M_R_fp32_shared[row_idx, col_idx] = T.cast(M_R_shared[row_idx, col_idx], T.float32)
                dM_fp32_shared[row_idx, col_idx] = T.cast(T.cast(dM_fragment[row_idx, col_idx], dtype), T.float32)
            T.gemm(M_R_fp32_shared, dM_fp32_shared, matrix_fragment, transpose_A=True, clear_accum=True)
            T.copy(matrix_fragment, matrix_fp32_workspace_shared)
            T.gemm(matrix_fp32_workspace_shared, M_R_fp32_shared, matrix_fragment, transpose_B=True, clear_accum=True)
            for row_idx, col_idx in T.Parallel(32, 32):
                dT_shared[row_idx, col_idx] = T.if_then_else(row_idx > col_idx, -matrix_fragment[row_idx, col_idx], 0.0)

            T.sync_threads()
            T.clear(dgamma_acc_fragment)
            for value_tile_idx in T.serial(T.ceildiv(value_dim, value_tile)):
                T.copy(
                    hat_V[chunk_idx, :, value_tile_idx * value_tile:value_tile_idx * value_tile + value_tile],
                    value_workspace_shared,
                )
                T.copy(
                    dgamma_E_shared[:, value_tile_idx * value_tile:value_tile_idx * value_tile + value_tile],
                    value_gradient_fragment,
                )
                T.gemm(dT_shared, value_workspace_shared, value_gradient_fragment, clear_accum=False)
                for row_idx, col_idx in T.Parallel(32, value_tile):
                    e = T.if_then_else(
                        value_tile_idx * value_tile + col_idx < value_dim,
                        T.cast(E[chunk_idx, row_idx, value_tile_idx * value_tile + col_idx], T.float32),
                        0.0,
                    )
                    dgamma_product_fragment[row_idx, col_idx] = value_gradient_fragment[row_idx, col_idx] * e
                    value_workspace_shared[row_idx, col_idx] = gamma_shared[row_idx] * e
                    if value_tile_idx * value_tile + col_idx < value_dim:
                        dE[chunk_idx, row_idx, value_tile_idx * value_tile + col_idx] = (
                            gamma_shared[row_idx] * value_gradient_fragment[row_idx, col_idx]
                        )
                T.reduce_sum(dgamma_product_fragment, dgamma_tile_fragment, dim=1)
                for row_idx in T.Parallel(32):
                    dgamma_acc_fragment[row_idx] += dgamma_tile_fragment[row_idx]
                T.copy(
                    dhat_V_shared[:, value_tile_idx * value_tile:value_tile_idx * value_tile + value_tile],
                    value_gradient_fragment,
                )
                T.gemm(dT_shared, value_workspace_shared, value_gradient_fragment, transpose_A=True, clear_accum=False)
                T.copy(
                    value_gradient_fragment,
                    dhat_V[chunk_idx, :, value_tile_idx * value_tile:value_tile_idx * value_tile + value_tile],
                )
            T.copy(dgamma_acc_fragment, dgamma[chunk_idx, :])

    return kernel


def right_operator_bwd(M_R, G_L, eta, gamma, E, hat_V, dPhi, dF_R, dN_Z, dhat_V):
    output_dtypes = G_L.dtype, E.dtype, hat_V.dtype
    M_R, G_L, E, hat_V, dPhi, dF_R, dN_Z, dhat_V = (
        tensor.to(torch.bfloat16)
        for tensor in (M_R, G_L, E, hat_V, dPhi, dF_R, dN_Z, dhat_V)
    )
    value_dim = hat_V.shape[-1]
    dG_L, deta, dgamma, dE, dhat_V_out = (
        torch.empty_like(tensor) for tensor in (G_L, eta, gamma, E, hat_V)
    )
    kernel = _build_right_operator_bwd_kernel(
        num_chunks=M_R.numel() // 1024,
        value_dim=value_dim,
    )
    kernel(
        M_R.view(-1, 32, 32), G_L.view(-1, 32, 32), eta.view(-1, 32), gamma.view(-1, 32),
        E.view(-1, 32, value_dim), hat_V.view(-1, 32, value_dim), dPhi.view(-1, 32, 32),
        dF_R.view(-1, 32, value_dim), dN_Z.view(-1, 32, value_dim), dhat_V.view(-1, 32, value_dim),
        dG_L.view(-1, 32, 32), deta.view(-1, 32), dgamma.view(-1, 32),
        dE.view(-1, 32, value_dim), dhat_V_out.view(-1, 32, value_dim),
    )
    return (
        dG_L.to(output_dtypes[0]), deta, dgamma,
        dE.to(output_dtypes[1]), dhat_V_out.to(output_dtypes[2]),
    )
