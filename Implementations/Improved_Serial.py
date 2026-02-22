import numpy as np
import sklearn
import scipy.linalg as LA
import matplotlib.pyplot as plt

from time import time
from sklearn.decomposition import SparseCoder


def kSVD(Y, T_0, k, num_iter, track_loss = True, verbose:int = 0, rng=42):
    loss = np.empty(num_iter)
    rng = np.random.default_rng(rng)

    # Initialize dictionary
    t0 = time()
    D = rng.standard_normal(size=(k, Y.shape[1]))
    D = (1 / LA.norm(D, ord=2, axis=1))[:, None] * D

    if verbose > 0:
        print(f'Initialization Time: {time() - t0}')
    
    for iter in range(num_iter):
        t0 = 0
        if verbose > 0:
            print(f'Iteration {iter}:')
            t0 = time()
        
        X = SparseCoder(dictionary = D, transform_n_nonzero_coefs = T_0).transform(Y)
        if verbose > 0:
            print(f'\tCoding Time: {time() - t0}')
        
        t0 = time()
        unused_atom = False
        for i in range(k):
            x_i = X[:, i]
            filter = (x_i != 0)
            x_i_R = x_i[filter]
            if x_i_R.shape[0] == 0:
                unused_atom=True
                continue
            
            E_k_R = Y[filter]
            for j in range(k):
                if j != i:
                    E_k_R -= X[filter, j][:, None] * D[j]

            U, S, Vh = LA.svd(E_k_R, full_matrices=False)

            X[filter, i] = U[:, 0] * S[0]
            D[i] = Vh[0]

        if verbose > 0:
            print(f'\tUpdate Time: {time() - t0}')
            print(f'\tUnused Atom: {unused_atom}')
        
        loss[iter] = LA.norm(Y - X @ D, ord='fro')

    return D, loss