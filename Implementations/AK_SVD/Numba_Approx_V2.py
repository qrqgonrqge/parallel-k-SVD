import numpy as np
import sklearn
import scipy.linalg as LA
import matplotlib.pyplot as plt

from time import time
from sklearn.decomposition import SparseCoder, sparse_encode
from sklearn.linear_model import orthogonal_mp_gram

import numba
from numba import njit

# OMP Helper Functions / Kernels
@njit
def proj(Q, D_k, J):
    B, _, D_1 = Q.shape
    dot = np.zeros((B, J), dtype = Q.dtype)
    for b in range(B):
        for j in range(J):
            for d in range(D_1):
                dot[b, j] += Q[b, j, d] * D_k[b, d]
    return dot

@njit
def ortho_comp(D_k, dot, Q, J):
    B, D_1 = D_k.shape
    q_j = D_k.copy()
    for b in range(B):
        for j in range(J):
            scale = dot[b, j]
            for d in range(D_1):
                q_j[b, d] -= Q[b, j, d] * scale 
    return q_j

@njit
def batch_dot(y, Q, j):
    B, D_1 = y.shape
    dot = np.zeros((B, ), dtype=Q.dtype)
    for b in range(B):
        for d in range(D_1):
            dot[b] += y[b, d] * Q[b, j, d]
    return dot

@njit
def res_update(r, Q, j, proj):
    B, _, D_1 = Q.shape
    # update = np.zeros((B, D_1), dtype=Q.dtype)
    for b in range(B):
        scale = proj[b]
        for d in range(D_1):
            r[b, d] -= Q[b, j, d] * scale

@njit
def norm(q_j):
    B, D_1 = q_j.shape
    norm = np.zeros((B, ), dtype=q_j.dtype)
    for b in range(B):
        for d in range(D_1):
            norm[b] += q_j[b, d] ** 2
        norm[b] = np.sqrt(norm[b])
    return norm

    
@njit
def q_j_and_R(Q, D_k, j, R):
    # Batched mat mul: (B, j, D.shape[1]) @ (B, D.shape[1], 1) -> (B, j, 1)
    # squeeze: (B, j, 1) -> (B, j)
    # dot = np.squeeze(Q[:, :j] @ D_k[..., np.newaxis], axis=-1)
    dot = proj(Q, D_k, j)
    
    R[:, 0:j, j] = dot
    
    # Orthogonalize
    # Batched mat mul: (B, 1, j) @ (B, j, D.shape[1]) -> (B, 1, D.shape[1])
    # squeeze: (B, 1, D.shape[1]) -> (B, D.shape[1])
    # subtraction: (B, D.shape[1]) - (B, D.shape[1]) -> (B, D.shape[1])
    # q_j = D_k - np.squeeze(dot[:, np.newaxis] @ Q[:, :j], axis = 1)
    return ortho_comp(D_k, dot, Q, j)

@njit
def stability_fix(q_j, global_dead_batches, atom_mask, k, Q, R, j):
                
    # norm, axis = 1: (B, D.shape[1]) -> (B,)
    # q_j_norm = np.linalg.norm(q_j, axis = 1)
    q_j_norm = norm(q_j)
    
    # STABILITY FIX
    # Instead of just Q[:, :, j] = q_j / q_j_norm[:, np.newaxis]
    dead_batches = (q_j_norm < 1e-15)

    global_dead_batches |= dead_batches
    q_j[dead_batches] = 0
    # atom_mask[dead_batches, k[dead_batches]] = 1
    for i, dead_batch in enumerate(dead_batches):
        if dead_batch:
            atom_mask[i, k[i]] = 1
    
    q_j_norm_safe = q_j_norm
    q_j_norm_safe[dead_batches] = 1 # Avoid division by zero
    Q[:, j] = q_j / q_j_norm_safe[:, np.newaxis]
    
    R[:, j, j] = q_j_norm

    
@njit
def OMP_GS_else(Q, D_k, j, R,
                 global_dead_batches, atom_mask, k):
    q_j = q_j_and_R(Q, D_k, j, R)
    stability_fix(q_j, global_dead_batches, atom_mask, k, Q, R, j)

@njit
def update_res_and_Q_T_y(y, Q, j, Q_T_y, r):
    # y_proj = np.sum(y * Q[:, j], axis = 1, dtype=float_dtype)
    y_proj = batch_dot(y, Q, j)
    
    Q_T_y[:, j] = y_proj

    # mul: (B, D.shape[1]) * (B, 1) -> (B, D.shape[1])
    # r -= Q[:, j] * y_proj[:, np.newaxis]
    res_update(r, Q, j, y_proj)

@njit
def OMP_GS_if(D_norm, k, R, D_k, Q):
    # Adv Indexing: (B,) -> (B, 1)
    D_k_norm = D_norm[k]
    R[:, 0, 0] = D_k_norm[:, 0]
    Q[:, 0] = D_k / D_k_norm

    
@njit
def GS(j, debug, y,
       D_norm, k, R, D_k, Q,
       global_dead_batches, atom_mask):
    if j == 0:
        OMP_GS_if(D_norm, k, R, D_k, Q)
        
    else:
        if debug:
            print(Q[:, :j].shape)
            print(D_k[..., np.newaxis].shape)
            
        OMP_GS_else(Q, D_k, j, R,
         global_dead_batches, atom_mask, k)
        
        if debug:            
            print(Q[:, :j+1].shape)
            print(y.shape)

@njit
def find_D_r(r, D, atom_mask):
    # Let B = batch_size
    # matmul: (B, D.shape[1]) @ (D.shape[1], D.shape[0]) -> (B, D.shape[0])
    D_r = r @ D.T
    D_r *= atom_mask
    # np.abs(D_r, out=D_r)
    return np.abs(D_r)

@njit
def atom_bookeeping(k, atom_mask, batch_idx, I, j, D):
    # atom_mask[batch_idx, k] = 0
    for idx in batch_idx:
        atom_mask[idx, k[idx]] = 0
    
    I[:, j] = k
    # D_I[:, j] = D[k]

    # Adv Indexing: (B,) -> (B, D.shape[1])
    D_k = D[k]

    return D_k

    
@njit
def unrolled_argmax(arr):
    n = arr.shape[0]
    if n == 0: return 0
    
    # We track 4 local maximums to allow the CPU to 
    # pipeline the instructions better.
    m1, m2, m3, m4 = 0, 0, 0, 0
    i1, i2, i3, i4 = 0, 0, 0, 0
    
    limit = (n // 4) * 4
    for i in range(0, limit, 4):
        v1, v2, v3, v4 = arr[i], arr[i+1], arr[i+2], arr[i+3]
        if v1 > m1: m1 = v1; i1 = i
        if v2 > m2: m2 = v2; i2 = i + 1
        if v3 > m3: m3 = v3; i3 = i + 2
        if v4 > m4: m4 = v4; i4 = i + 3
            
    # Final reduction
    best_val = m1
    idx = i1
    if m2 > best_val or (m2 == best_val and i2 < idx): 
        best_val = m2; idx = i2
    if m3 > best_val or (m3 == best_val and i3 < idx): 
        best_val = m3; idx = i3
    if m4 > best_val or (m4 == best_val and i4 < idx): 
        best_val = m4; idx = i4
    
    # Cleanup for remainder
    for i in range(limit, n):
        if arr[i] > best_val:
            best_val = arr[i]; idx = i
            
    return idx

@njit
def best_atom(D_r, dtype):
    B = D_r.shape[0]
    k = np.empty((B, ), dtype = dtype)
    for b in range(B):
        k[b] = unrolled_argmax(D_r[b])
    return k

    
@njit
def OMP_inner_loop(debug, i,
                   global_dead_batches, j_stop, 
                   r, D, atom_mask,
                   int_dtype,
                   batch_idx, I, 
                   y, D_norm, R, Q,
                   T_0, 
                   Q_T_y):
    for j in range(T_0):
        if debug:
            print(f'Batch {i}, Iteration {j}')
            print(f'r shape: {r.shape}')
    
        # Check if any batch is alive otherwise, exit loop
        if np.all(global_dead_batches):
            j_stop = j
            break
    
        # Find best dictionary atom  
    
        # # Let B = batch_size
        # # matmul: (B, D.shape[1]) @ (D.shape[1], D.shape[0]) -> (B, D.shape[0])
        # D_r = r @ D.T
        # D_r *= atom_mask
        # np.abs(D_r, out=D_r)
        D_r = find_D_r(r, D, atom_mask)
    
        
        # # max, axis = 1: (B, D.shape[0]) -> (B,)
        # k = np.argmax(D_r, axis = 1)   #488.5
        k = best_atom(D_r, int_dtype)   #1205.0
        
        # atom_mask[batch_idx, k] = 0
        
        # I[:, j] = k
        # # D_I[:, j] = D[k]

        # # Adv Indexing: (B,) -> (B, D.shape[1])
        # D_k = D[k]
        D_k = atom_bookeeping(k, atom_mask, batch_idx, I, j, D)
    
        # Gram-Schmidt
        GS(j, debug, y,
           D_norm, k, R, D_k, Q,
           global_dead_batches, atom_mask)
        
    
        if debug and j == (T_0 - 1):
            print(f'y dtype: {y.dtype}')
            print(f'Q dtype: {Q.dtype}')
            
    
        # y_proj = batch_dot5(y, Q, j)
        # Q_T_y[:, j] = y_proj
        # res_update4(r, Q, j, y_proj)
        update_res_and_Q_T_y(y, Q, j, Q_T_y, r)
            
        if debug:
            # print(gamma.shape)
            # print(f'D_I shape: {D_I[:, :j+1].shape}')
            print(f'y shape: {y.shape}')
        # est = np.squeeze(gamma[:, np.newaxis] @ D_I[:, :j+1], axis=1)
            # print(f'est shape: {est.shape}')
        # r = y - est
        # r = y_ortho_proj
    
            print(f'r shape: {r.shape}')
            # print(f'error = {np.sum(r * r, axis = -1)}')

    
# OMP Entry Point / Orchestrator
@njit
def OMP(Y, T_0, D, batch_size = 1, rng=42, debug=False, float_dtype=np.float32):
    int_dtype = np.int32
    
    X = np.zeros((Y.shape[0], D.shape[0])[::-1], dtype=float_dtype).T
    # D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D_norm = np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    
    splits = np.arange(0, Y.shape[0], step=batch_size, dtype=int_dtype)

    # Y_batches = np.split(Y, splits)[1:]

    row_idx = np.arange(Y.shape[0], dtype=int_dtype)
    # row_batch_idx = np.split(row_idx, splits)[1:]
    
    batch_idx = np.arange(batch_size, dtype=int_dtype)

    res = Y.copy()
    
    # for i, y in enumerate(Y_batches):
    for i, split in enumerate(splits):

        start_idx = split
        end_idx = min(start_idx + batch_size, Y.shape[0])
        batch_size = end_idx - start_idx
        batch_idx = batch_idx[:batch_size]
        # if i == (len(Y_batches) - 1):
        #     batch_size = np.arange(splits[i], len(Y), dtype=int_dtype).shape[0]
        #     batch_idx = batch_idx[:batch_size]
        #     if debug:
        #         print(f'batch_size = {batch_size}')
        y = Y[start_idx:end_idx]
                
        I = np.empty((batch_size, T_0), dtype=int_dtype)
        D_I = np.zeros((batch_size, T_0, D.shape[1]), dtype=float_dtype)

        # Deep copy to not overwrite Y
        # r = y.copy()   # (batch_size, D.shape[1])
        # r = y.astype(float_dtype)
        r = res[start_idx:end_idx]
        gamma = 0
        
        # Q = np.empty((batch_size, D.shape[1], T_0))
        Q = np.empty((batch_size, T_0, D.shape[1]), dtype=float_dtype)
        
        Q_T_y = np.empty((batch_size, T_0)[::-1], dtype=float_dtype).T
        # R = np.empty((batch_size, T_0, T_0))
        R = np.zeros((batch_size, T_0, T_0)[::-1], dtype=float_dtype).T
        

        # Create a mask to ensure duplicates aren't selected
        atom_mask = np.ones((batch_size, D.shape[0]), dtype=float_dtype)

        global_dead_batches = np.zeros(batch_size, dtype=np.bool_)
        j_stop = T_0
        
        OMP_inner_loop(debug, i, 
               global_dead_batches, j_stop, 
               r, D, atom_mask,
               int_dtype,
               batch_idx, I, 
               y, D_norm, R, Q,
               T_0, 
               Q_T_y)

        # rows = row_batch_idx[i]
        rows = row_idx[start_idx:end_idx]

        gamma = np.empty((batch_size, j_stop), dtype=float_dtype)
        for n in range(1, j_stop+1):
            idx = j_stop - n
            gamma_idx = Q_T_y[:, idx] / R[:, idx, idx]
            gamma[:, idx] = gamma_idx
            for i in range(idx):
                Q_T_y[:, i] -= gamma_idx * R[:, i, idx]

        M, _ = I.shape
        for m in range(M):
            for n in range(j_stop):
                X[rows[m], I[m, n]] = gamma[m, n]
        
        if debug:
            print()
    return X


# kSVD routines
@njit
def read_filter(E_k_R, filter, X_filter_i, D, i):
    return E_k_R[filter] + X_filter_i[:, np.newaxis] * D[i]

@njit
def write_filter(E_k_R, filter, E_k_R_filter, X_filter_i, D, i):
    E_k_R[filter] = E_k_R_filter - X_filter_i[:, np.newaxis] * D[i]


@njit
def dict_update(X_filter_i, E_k_R_filter, D, i, X, filter):
    # U, S, Vh = np.linalg.svd(E_k_R[filter], full_matrices=False)
    # X[filter, i] = U[:, 0] * S[0]
    # D[i] = Vh[0]
    V_est = X_filter_i @ E_k_R_filter
    V_est /= np.linalg.norm(V_est)
    D[i] = V_est

    X_filter_i = E_k_R_filter @ V_est
    X[filter, i] = X_filter_i

    return X_filter_i


@njit
def kSVD_inner(k, filter_bool, unused_atom, E_k_R, dtype, Y, D, X, verbose):
    for i in range(k):
        filter_bool_i = filter_bool[:, i]
        
        fil = np.flatnonzero(filter_bool_i)
        # print('a')
        if fil.shape[0] == 0:
            unused_atom=True
            # sum, axis=1: (Y.shape[0], Y.shape[1]) -> (Y.shape[0], )
            atom_error = np.sum(E_k_R ** 2, axis=1, dtype=dtype)
            atom_idx = np.argmax(atom_error)

            D_i_new = Y[atom_idx]
            D_i_new /= np.linalg.norm(D_i_new)
            D[i] = D_i_new

            X[atom_idx, i] = 1
            E_k_R[atom_idx] = 0
            if verbose > 0:
                print(f'Replaced dict atom {i} with data point {atom_idx}')
            continue
        
        # print('b')
        X_filter_i = X[fil, i]
        # E_k_R_filter = E_k_R[filter] + X_filter_i[:, np.newaxis] * D[i]
        E_k_R_filter = read_filter(E_k_R, fil, X_filter_i, D, i)
        # E_k_R_filter = read_filter2(E_k_R, filter, X_filter_i, D, i)
        # print('c')
        X_filter_i = dict_update(X_filter_i, E_k_R_filter, D, i, X, fil)
        # print('d')
        # E_k_R[filter] = E_k_R_filter - X_filter_i[:, np.newaxis] * D[i]
        write_filter(E_k_R, fil, E_k_R_filter, X_filter_i, D, i)
        # write_filter2(E_k_R, filter, E_k_R_filter, X_filter_i, D, i)
    return unused_atom


@njit
def kSVD_outer(num_iter, verbose,
               Y, T_0, D,
               batch_size, rng, dtype,
               k, loss):
    for iter in range(num_iter):
        t0 = 0
        if verbose > 0:
            print(f'Iteration {iter}:')
            # t0 = time()
        
        X = OMP(Y, T_0, D, 
                batch_size = batch_size, 
                rng=rng, 
                debug=(verbose > 0),
                float_dtype=dtype)
        # if verbose > 0:
            # print(f'\tCoding Time: {time() - t0}')
        
        # t0 = time()
        unused_atom = False
        filter_bool = (X != 0)
        # matmul: (Y.shape[0], k) @ (k, Y.shape[1]) -> (Y.shape[0], Y.shape[1]) 
        E_k_R = Y - X @ D
        # E_k_R = np.subtract(Y, X @ D, dtype=dtype)
        
        unused_atom = kSVD_inner(k, filter_bool, unused_atom, E_k_R, dtype, Y, D, X, verbose)

        if verbose > 0:
            # print(f'\tUpdate Time: {time() - t0}')
            print(f'\tUnused Atom Detected: {unused_atom}')
        
        # loss[iter] = LA.norm(E_k_R, ord='fro')
        loss[iter] = np.linalg.norm(E_k_R)


# kSVD entry point
def kSVD(Y, T_0, k, num_iter, 
         batch_size = 1, 
         track_loss = True, 
         verbose:int = 0, 
         rng = 42,
         dtype = np.float32):
    loss = np.empty(num_iter, dtype=dtype)
    rng = np.random.default_rng(rng)

    if dtype != np.float64 and dtype != np.float32:
        print('Only FP32 or FP64 is supported!')
        print('Setting the dtype to FP32...')
        dtype = np.float32
    
    # Initialize dictionary
    t0 = time()
    D = rng.standard_normal(size=(k, Y.shape[1]), dtype=dtype)
    # D /= LA.norm(D, ord=2, axis=1)[:, np.newaxis]
    D /= np.linalg.norm(D, ord=2, axis=1, keepdims=True)

    if verbose > 0:
        print(f'Initialization Time: {time() - t0}')

    if Y.dtype != dtype:
        if verbose > 0:
            print("dtype arg doesn't match Y dtype!")
            print('Creating a copy of Y with the specified dtype...')
        Y = Y.astype(dtype)
    
    kSVD_outer(num_iter, verbose,
               Y, T_0, D,
               batch_size, rng, dtype,
               k, loss)

    return D, loss