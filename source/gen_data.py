#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import sys
import numpy as np
from scipy.sparse import coo_matrix


def gen_dupl_mat(S, n_by_N, prop_dupli=1, rand_seed=1):
    """
    Generate observed matrix A, and duplication count C, and true assignment Z
    from ground truth matrix S
    """
    (N, m) = S.shape
    assert(N == m)

    if n_by_N >= 1:
        print("n_by_N({}) must be < 1. Setting to 1/2.".format(n_by_N))
    n_ = int(N * n_by_N)

    if prop_dupli > 1:
        print("prop_dupli ({}) must be <= 1. Setting to 1.".format(prop_dupli))
    n_dupli = int(n_ * prop_dupli)

    # For reproductibility
    np.random.seed(rand_seed)
    # Just in case...
    n_ = max(n_, 1)
    n_dupli = max(n_dupli, 1)

    n_add = N - n_
    shuffled_N = np.random.permutation(N)
    zis = np.zeros(0, dtype=int)
    zjs = np.zeros(0, dtype=int)

    # Add one element to each bin
    zis = np.append(zis, np.arange(n_))
    zjs = np.append(zjs, shuffled_N[:n_])
    # Split the copy number surplus among the n_ nodes
    which_selected = np.random.randint(0, high=n_, size=n_add)
    zis = np.append(zis, which_selected)
    zjs = np.append(zjs, shuffled_N[n_:])
    # Build assignment matrix
    Z = coo_matrix((np.ones(N), (zis, zjs)), shape=(n_, N), dtype=int)
    # C = np.sum(Z, axis=1)
    C = Z @ np.ones(N, dtype=int)
    A = Z @ S @ Z.T

    return(Z, A, C)

def gen_chr_mat(n, n_chr):

    if n_chr == 1:
        X = np.ones((n, n))
        return(X)
    x_flat = np.zeros(n**2)
    size_chr = n // n_chr
    for k in range(n_chr):
        sub_idxs = np.arange(size_chr) + k * size_chr
        n_clus = len(sub_idxs)
        iis = np.repeat(sub_idxs, n_clus)
        jjs = np.tile(sub_idxs, n_clus)
        idxs = np.ravel_multi_index((iis, jjs), (n, n))
        x_flat[idxs] = 1
    sub_idxs = np.arange((n_chr) * size_chr, n)
    n_clus = len(sub_idxs)
    iis = np.repeat(sub_idxs, n_clus)
    jjs = np.tile(sub_idxs, n_clus)
    idxs = np.ravel_multi_index((iis, jjs), (n, n))
    x_flat[idxs] = 1
    X = np.reshape(x_flat, (n, n))
    return X





if __name__ == 'main':

    from mdso import SimilarityMatrix
    import matplotlib.pyplot as plt

    n = 200
    type_noise = 'gaussian'
    ampl_noise = 0.5
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    S = data_gen.sim_matrix

    n_by_N = 0.7
    prop_dupli = 0.9
    rand_seed = 1

    (Z, A, C) = gen_dupl_mat(S, n_by_N, prop_dupli=prop_dupli,
                             rand_seed=rand_seed)
    plt.matshow(S)
    plt.show()
