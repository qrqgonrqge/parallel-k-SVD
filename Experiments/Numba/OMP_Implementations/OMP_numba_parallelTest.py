import numba
import numpy as np
from numba import jit, njit, prange

def OMP_numba_test_nojit(Y,T_0,D,batch_size = 1, rng=42, debug=False):
    # print('Hello from numba!')
    X = np.zeros((Y.shape[0], D.shape[0])[::-1]).T

    # D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D_norm = np.array([np.linalg.norm(D[i]) for i in range(D.shape[0])])[:, np.newaxis]
    if debug:
        print(D.shape, D_norm.shape)
        # print(D_norm)
        
    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)

@njit(parallel=False)
def OMP_numba_test_paraFalse(Y,T_0,D,batch_size = 1, rng=42, debug=False):
    # print('Hello from numba!')
    X = np.zeros((Y.shape[0], D.shape[0])[::-1]).T

    # D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D_norm = np.array([np.linalg.norm(D[i]) for i in range(D.shape[0])])[:, np.newaxis]
    if debug:
        print(D.shape, D_norm.shape)
        # print(D_norm)
        
    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)

@njit(parallel=True)
def OMP_numba_test_paraTrue(Y,T_0,D,batch_size = 1, rng=42, debug=False):
    # print('Hello from numba!')
    X = np.zeros((Y.shape[0], D.shape[0])[::-1]).T

    # D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D_norm = np.array([np.linalg.norm(D[i]) for i in range(D.shape[0])])[:, np.newaxis]
    if debug:
        print(D.shape, D_norm.shape)
        # print(D_norm)
        
    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)

@njit(parallel=True)
def OMP_numba_test_paraTrue_wPrange(Y,T_0,D,batch_size = 1, rng=42, debug=False):
    # print('Hello from numba!')
    X = np.zeros((Y.shape[0], D.shape[0])[::-1]).T

    # D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D_norm = np.array([np.linalg.norm(D[i]) for i in prange(D.shape[0])])[:, np.newaxis]
    if debug:
        print(D.shape, D_norm.shape)
        # print(D_norm)
        
    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)



# @njit(parallel=True)
def OMP_numba_1(Y, T_0, D, batch_size = 1, rng=42, debug=False):

    X = np.zeros((Y.shape[0], D.shape[0]), order='F')
    D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    print(D.shape, D_norm.shape)

    
    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)
    
    print(len(Y_batches))
    print(Y_batches[0].shape)
    for i, y in enumerate(Y_batches):
        
        if i == (len(Y_batches) - 1):
            batch_size = np.arange(splits[i], len(Y)).shape[0]
            batch_idx = batch_idx[:batch_size]
            if debug:
                print(f'batch_size = {batch_size}')
                
        I = np.empty((batch_size, T_0), dtype=np.int32)
        D_I = np.zeros((batch_size, T_0, D.shape[1]))

        # Deep copy to not overwrite Y
        r = y.copy()   # (batch_size, D.shape[1])
        gamma = 0
        
        Q = np.empty((batch_size, D.shape[1], T_0))
        Q_T_y = np.empty((batch_size, T_0), order='F')
        # R = np.empty((batch_size, T_0, T_0))
        R = np.zeros((batch_size, T_0, T_0), order='F')
        

        # Create a mask to ensure duplicates aren't selected
        atom_mask = np.ones((batch_size, D.shape[0]))

        global_dead_batches = np.zeros(batch_size, dtype=bool)
        j_stop = T_0
        
        for j in range(T_0):
            if debug:
                print(f'Batch {i}, Iteration {j}')
                print(f'r shape: {r.shape}')

            # Check if any batch is alive otherwise, exit loop
            if np.all(global_dead_batches):
                j_stop = j
                break

            # Find best dictionary atom  

            # Let B = batch_size
            # matmul: (B, D.shape[1]) @ (D.shape[1], D.shape[0]) -> (B, D.shape[0])
            D_r = r @ D.T
            D_r *= atom_mask
            np.abs(D_r, out=D_r)
            
            # max, axis = 1: (B, D.shape[0]) -> (B,)
            k = np.argmax(D_r, axis = 1)

            atom_mask[batch_idx, k] = 0
            
            I[:, j] = k
            # D_I[:, j] = D[k]

            # Adv Indexing: (B,) -> (B, D.shape[1])
            D_k = D[k]

            # Gram-Schmidt
            if j == 0:
                # Adv Indexing: (B,) -> (B, 1)
                D_k_norm = D_norm[k]
                R[:, 0, 0] = D_k_norm[:, 0]
                Q[:, :, 0] = D_k / D_k_norm
                
                # if debug:
                    # print(D_k_norm)
                    # print(np.sum(Q[:, :, 0] ** 2, axis = 1))
                    # print(R[:, 0, 0])

                # matmul: (B, 1, D.shape[1]) @ (B, D.shape[1], 1) -> (B, 1, 1) 
                # y_proj = y[:, np.newaxis] @ Q[:, :, :1]
                # y_proj = np.sum(y * Q[:, :, 0], axis = 1)

                # squeeze: (B, 1, 1) -> (B, 1)
                # gamma = np.squeeze(y_proj, axis=1) 
                # gamma /= D_k_norm

                # transpose (0, 2, 1): (B, 1, 1) -> (B, 1, 1)
                # y_proj = np.transpose(y_proj, (0, 2, 1))

                # matmul: (B, D.shape[1], 1) @ (B, 1, 1)  -> (B, D.shape[1], 1)
                # squeeze: (B, D.shape[1], 1) -> (B, D.shape[1])
                # r -= np.squeeze(Q[:, :, :1] @ y_proj, axis=-1)
                # r -= Q[:, :, 0] * y_proj[:, np.newaxis]
                
            else:
                if debug:
                    print(np.transpose(Q[:, :, :j], (0, 2, 1)).shape)
                    print(D_k[..., np.newaxis].shape)
                    
                # Batched mat mul: (B, 1, D.shape[1]) @ (B, D.shape[1], j) -> (B, 1, j)
                # squeeze: (B, 1, j) -> (B, j)
                dot = np.squeeze(D_k[:, np.newaxis] @ Q[:, :, :j], axis=1)
                R[:, 0:j, j] = dot

                # Orthogonalize
                # Batched mat mul: (B, D.shape[1], j) @ (B, j, 1) -> (B, D.shape[1], 1)
                # squeeze: (B, D.shape[1], 1) -> (B, D.shape[1])
                # subtraction: (B, D.shape[1]) - (B, D.shape[1]) -> (B, D.shape[1])
                q_j = D_k - np.squeeze(Q[:, :, :j] @ dot[..., np.newaxis], axis = -1)
                
                # norm, axis = 1: (B, D.shape[1]) -> (B,)
                q_j_norm = np.linalg.norm(q_j, axis = 1)

                # STABILITY FIX
                # Instead of just Q[:, :, j] = q_j / q_j_norm[:, np.newaxis]
                dead_batches = (q_j_norm < 1e-15)

                global_dead_batches |= dead_batches
                q_j[dead_batches] = 0
                atom_mask[dead_batches, k[dead_batches]] = 1
                
                q_j_norm_safe = q_j_norm
                q_j_norm_safe[dead_batches] = 1 # Avoid division by zero
                Q[:, :, j] = q_j / q_j_norm_safe[:, np.newaxis]
                
                R[:, j, j] = q_j_norm
                # Q[:, :, j] = q_j / q_j_norm[:, np.newaxis]
                
                if debug:            
                    print(Q[:, :, :j+1].shape)
                    print(y.shape)

                # matmul: (B, 1, D.shape[1]) @ (B, D.shape[1], 1) -> (B, 1, 1) 
                # transpose (0, 2, 1): (B, 1, 1) -> (B, 1, 1)
                # Q_T_y = np.transpose(y[:, np.newaxis] @ Q[:, :, :j+1], (0, 2, 1))
                # y_proj = np.transpose(y[:, np.newaxis] @ Q[:, :, j:j+1], (0, 2, 1))
                # y_proj = np.sum(y * Q[:, :, j], axis = 1)

                # matmul: (B, D.shape[1], 1) @ (B, 1, 1)  -> (B, D.shape[1], 1)
                # squeeze: (B, D.shape[1], 1) -> (B, D.shape[1])
                # r -= np.squeeze(Q[:, :, j:j+1] @ y_proj, axis=-1)
                # r -= Q[:, :, j] * y_proj[:, np.newaxis]
                
                # gamma = LA.solve_triangular(R[:, :j+1, :j+1], 
                #                         Q_T_y, 
                #                         overwrite_b = True,
                #                         check_finite = False)

                # R_temp = R[:, :j+1, :j+1]
                # for b in range(batch_size):
                #     # Adding a tiny jitter to the diagonal prevents the singular error 
                #     # if you insist on linalg.solve
                #     if dead_batches[b]:
                #         R_temp[b, j, j] = 1 
                
                # gamma = np.linalg.solve(R[:, :j+1, :j+1], Q_T_y)
                # gamma = np.linalg.solve(R_temp, Q_T_y)
                
                # if debug:
                #     print(f'original gamma shape: {gamma.shape}')
                # gamma = np.squeeze(gamma, axis=-1)

            # sum, axis=1: (B, D.shape[1]) -> (B,)
            y_proj = np.sum(y * Q[:, :, j], axis = 1)
            Q_T_y[:, j] = y_proj

            # mul: (B, D.shape[1]) * (B, 1) -> (B, D.shape[1])
            r -= Q[:, :, j] * y_proj[:, np.newaxis]

                
            if debug:
                # print(gamma.shape)
                # print(f'D_I shape: {D_I[:, :j+1].shape}')
                print(f'y shape: {y.shape}')
                est = np.squeeze(gamma[:, np.newaxis] @ D_I[:, :j+1], axis=1)
                print(f'est shape: {est.shape}')
            # r = y - est
            # r = y_ortho_proj

                print(f'r shape: {r.shape}')
                print(f'error = {np.sum(r * r, axis = -1)}')

        # matmul: (B, 1, D.shape[1]) @ (B, D.shape[1], j_stop) -> (B, 1, j_stop) 
        # transpose (0, 2, 1): (B, 1, j_stop) -> (B, j_stop, 1)
        # Q_T_y = np.transpose(y[:, np.newaxis] @ Q[:, :, :j_stop+1], (0, 2, 1))
        gamma = np.linalg.solve(R[:, :j_stop, :j_stop], Q_T_y[:, :j_stop, np.newaxis])
        gamma = np.squeeze(gamma, axis=-1)
        

        rows = 0
        cols = I[:, :j_stop]
        if i == (len(Y_batches) - 1):
            if debug:
                # print(splits[i].dtype)
                # print(I.dtype)
                print(f'gamma: {gamma}')
                print(f'gamma shape: {gamma.shape}')
                print(f'k shape: {k.shape}')
                print(f'I shape: {I.shape}')
            rows = np.arange(splits[i], len(Y))[:, np.newaxis]
            
        elif i == 0:
            rows = np.arange(0, splits[1])[:, np.newaxis]
        else:
            rows = np.arange(splits[i], splits[i+1])[:, np.newaxis]

        X[rows, cols] = gamma

        if debug:
            print()
    return X

from sklearn.datasets import make_regression
X, y = make_regression(n_samples = 50, n_features = 300, n_targets = 20_000, noise=4, random_state=0)
OMP_numba_1(y.T, 1, X.T, batch_size=1)

# OMP_numba_test(y.T, 1, X.T, batch_size=1, debug=True)

import timeit
N = 10
print('OMP_numba_nojit time:', timeit.timeit(lambda: OMP_numba_test_nojit(y.T, 1, X.T, batch_size=1), number=N)/N, 's')
print('OMP_numba_test_paraFalse time:', timeit.timeit(lambda: OMP_numba_test_paraFalse(y.T, 1, X.T, batch_size=1), number=N)/N, 's')
print('OMP_numba_test_paraTrue time:', timeit.timeit(lambda: OMP_numba_test_paraTrue(y.T, 1, X.T, batch_size=1), number=N)/N, 's')
print('OMP_numba_test_paraTrue_wPrange time:', timeit.timeit(lambda: OMP_numba_test_paraTrue_wPrange(y.T, 1, X.T, batch_size=1), number=N)/N, 's')