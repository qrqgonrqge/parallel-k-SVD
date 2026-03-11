import numpy as np
import sklearn
import scipy.linalg as LA
import matplotlib.pyplot as plt

from time import time
from sklearn.decomposition import SparseCoder, sparse_encode
from sklearn.linear_model import orthogonal_mp_gram


def OMP(Y, T_0, D, batch_size = 1, rng=42, debug=False):

    X = np.zeros((Y.shape[0], D.shape[0]))
    D_norm = np.linalg.norm(D, axis=1, keepdims=True)

    splits = np.arange(0, Y.shape[0], step=batch_size)
    Y_batches = np.split(Y, splits)[1:]
    batch_idx = np.arange(batch_size)
    
    for i, y in enumerate(Y_batches):
        
        if i == (len(Y_batches) - 1):
            batch_size = np.arange(splits[i], len(Y)).shape[0]
            batch_idx = batch_idx[:batch_size]
            if debug:
                print(f'batch_size = {batch_size}')
                
        I = np.empty((batch_size, T_0), dtype=np.int32)
        D_I = np.zeros((batch_size, T_0, D.shape[1]))
        r = y   # (batch_size, D.shape[1])
        gamma = 0
        
        Q = np.empty((batch_size, D.shape[1], T_0))
        # R = np.empty((batch_size, T_0, T_0))
        R = np.zeros((batch_size, T_0, T_0))
        

        # Create a mask to ensure duplicates aren't selected
        atom_mask = np.ones((batch_size, D.shape[0]))
        # active = np.ones(batch_size, dtype=bool)
        
        for j in range(T_0):
            if debug:
                print(f'Batch {i}, Iteration {j}')
                print(f'r shape: {r.shape}')

            # Find best dictionary atom  
            # matmul: (batch_size, D.shape[1]) @ (D.shape[1], D.shape[0]) -> (batch_size, D.shape[0])
            # D_r = np.abs(r @ D.T) * atom_mask
            D_r = r @ D.T
            D_r = np.abs(D_r)
            D_r *= atom_mask
            
            # max, axis = 1: (batch_size, D.shape[0]) -> (batch_size,)
            k = np.argmax(D_r, axis = 1)

            atom_mask[batch_idx, k] = 0
            
            I[:, j] = k
            D_I[:, j] = D[k]

            # Gram-Schmidt
            if j == 0:
                D_k_norm = D_norm[k]
                R[:, 0, 0] = D_k_norm[:, 0]
                Q[:, :, 0] = D[k] / D_k_norm
                
                # if debug:
                    # print(D_k_norm)
                    # print(np.sum(Q[:, :, 0] ** 2, axis = 1))
                    # print(R[:, 0, 0])

                gamma = np.squeeze(y[:, np.newaxis] @ Q[:, :, :1], axis=1) 
                gamma /= D_k_norm
                
            else:
                if debug:
                    print(np.transpose(Q[:, :, :j], (0, 2, 1)).shape)
                    print(D[k][..., np.newaxis].shape)
                    
                # Let B = batch_size
                # Batched mat mul: (B, 1, D.shape[1]) @ (B, D.shape[1], j) -> (B, 1, j)
                # squeeze: (B, 1, j) -> (B, j)
                dot = np.squeeze(D[k][:, np.newaxis] @ Q[:, :, :j], axis=1)
                R[:, 0:j, j] = dot

                # Orthogonalize
                # Batched mat mul: (B, D.shape[1], j) @ (B, j, 1) -> (B, D.shape[1], 1)
                # squeeze: (B, D.shape[1], 1) -> (B, D.shape[1])
                # subtraction: (B, D.shape[1]) - (B, D.shape[1]) -> (B, D.shape[1])
                q_j = D[k] - np.squeeze(Q[:, :, :j] @ dot[..., np.newaxis], axis = -1)
                # norm, axis = 1: (B, D.shape[1]) -> (B,)
                q_j_norm = np.linalg.norm(q_j, axis = 1)

                # STABILITY FIX
                # Instead of just Q[:, :, j] = q_j / q_j_norm[:, np.newaxis]
                dead_batches = (q_j_norm < 1e-14)
                q_j[dead_batches] = 0
                atom_mask[batch_idx[dead_batches], k[dead_batches]] = 1
                
                q_j_norm_safe = q_j_norm.copy()
                q_j_norm_safe[dead_batches] = 1 # Avoid division by zero
                Q[:, :, j] = q_j / q_j_norm_safe[:, np.newaxis]
                
                R[:, j, j] = q_j_norm
                # Q[:, :, j] = q_j / q_j_norm[:, np.newaxis]
                
                if debug:            
                    print(Q[:, :, :j+1].shape)
                    print(y.shape)
                Q_T_y = np.transpose(y[:, np.newaxis] @ Q[:, :, :j+1], (0, 2, 1))
                # gamma = LA.solve_triangular(R[:, :j+1, :j+1], 
                #                         Q_T_y, 
                #                         overwrite_b = True,
                #                         check_finite = False)

                R_temp = R[:, :j+1, :j+1].copy()
                for b in range(batch_size):
                    # Adding a tiny jitter to the diagonal prevents the singular error 
                    # if you insist on linalg.solve
                    if dead_batches[b]:
                        R_temp[b, j, j] = 1 
                
                # gamma = np.linalg.solve(R[:, :j+1, :j+1], Q_T_y)
                gamma = np.linalg.solve(R_temp + np.eye(j+1) * 1e-15, Q_T_y)
                gamma[dead_batches, j] = 0
                
                if debug:
                    print(f'original gamma shape: {gamma.shape}')
                gamma = np.squeeze(gamma, axis=-1)
                

            if debug:
                print(gamma.shape)
                print(f'D_I shape: {D_I[:, :j+1].shape}')
                print(f'y shape: {y.shape}')
            est = np.squeeze(gamma[:, np.newaxis] @ D_I[:, :j+1], axis=1)
            if debug:
                print(f'est shape: {est.shape}')
            r = y - est

            if debug:
                print(f'r shape: {r.shape}')
            if debug:
                print(f'error = {np.sum(r * r, axis = -1)}')

        if i == (len(Y_batches) - 1):
            if debug:
                # print(splits[i].dtype)
                # print(I.dtype)
                print(f'gamma: {gamma}')
                print(f'gamma shape: {gamma.shape}')
                print(f'k shape: {k.shape}')
                print(f'I shape: {I.shape}')

            X[np.arange(splits[i], len(Y))[:, np.newaxis], I] = gamma
            
        elif i == 0:
            X[np.arange(0, splits[1])[:, np.newaxis], I] = gamma
        else:
            X[np.arange(splits[i], splits[i+1])[:, np.newaxis], I] = gamma

        if debug:
            print()
    return X



def kSVD_improved(Y, T_0, k, num_iter, batch_size = 1, track_loss = True, verbose:int = 0, rng=42):
    loss = np.empty(num_iter)
    rng = np.random.default_rng(rng)

    # Initialize dictionary
    t0 = time()
    D = rng.standard_normal(size=(k, Y.shape[1]))
    D /= LA.norm(D, ord=2, axis=1)[:, np.newaxis]

    if verbose > 0:
        print(f'Initialization Time: {time() - t0}')
    
    for iter in range(num_iter):
        t0 = 0
        if verbose > 0:
            print(f'Iteration {iter}:')
            t0 = time()
        
        # gram = D @ D.T
        # cov = D @ Y.T
        # X = orthogonal_mp_gram(
        #     Gram=gram, 
        #     Xy=cov, 
        #     n_nonzero_coefs=T_0, 
        #     copy_Gram=False, 
        #     copy_Xy=False
        # ).T # Transpose to match X shape
        X = OMP(Y, T_0, D, batch_size = batch_size, rng=rng, debug=(verbose > 0))
        # X = np.asfortranarray(X)
        if verbose > 0:
            print(f'\tCoding Time: {time() - t0}')
        
        t0 = time()
        unused_atom = False
        # XD = X @ D
        filter_bool = (X != 0)
        E_k_R = Y - X @ D
        for i in range(k):
            # x_i = X[:, i]
            filter_bool_i = filter_bool[:, i]
            
            # filter = np.flatnonzero(x_i)
            # filter = np.nonzero(x_i)[0]
            filter = np.flatnonzero(filter_bool_i)
            
            # x_i_R = x_i[filter]
            
            # if x_i_R.shape[0] == 0:
            if filter.shape[0] == 0:
                unused_atom=True
                continue
            
            # res = X[filter, i][:, np.newaxis] * D[i]
            # XD[filter] -= res
            # E_k_R = Y[filter] - XD[filter]
            E_k_R[filter] += X[filter, i][:, np.newaxis] * D[i]

            # U, S, Vh = LA.svd(E_k_R, full_matrices=False)
            # U, S, Vh = LA.svd(E_k_R[filter], full_matrices=False)
            U, S, Vh = np.linalg.svd(E_k_R[filter], full_matrices=False)

            X[filter, i] = U[:, 0] * S[0]
            D[i] = Vh[0]

            # XD[filter] += X[filter, i][:, np.newaxis] * D[i]
            E_k_R[filter] -= X[filter, i][:, np.newaxis] * D[i]

        if verbose > 0:
            print(f'\tUpdate Time: {time() - t0}')
            print(f'\tUnused Atom: {unused_atom}')
        
        # loss[iter] = LA.norm(Y - XD, ord='fro')
        loss[iter] = LA.norm(E_k_R, ord='fro')

    return D, loss