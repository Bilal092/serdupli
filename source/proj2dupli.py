#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import sys
import numpy as np
from scipy.sparse import find


def one_proj_sorted(s_vec, a_val, u_b=None):
    """
    Finds x = argmin |x - s_vec|, such that x_i >= 0 and sum_i x_i = a.
    """
    p = len(s_vec)
    s_sum = np.sum(s_vec)
    perm = np.argsort(-s_vec)
    invperm = np.argsort(perm)
    s = s_vec[perm]

    y = s + (a_val - s_sum) / p

    if a_val > s_sum:
        # y = s + (a_val - s_sum) / p
        if u_b is not None:
            if y[0] > u_b:
                above_bound = np.where(y > u_b)[0]
                if len(above_bound) == p:
                    warnings.warn("pb. infeasible with upper bound {}".format(
                        u_b
                    ))
                    y[above_bound] = u_b
                else:
                    y[above_bound] = u_b
                    y[above_bound[-1]+1:] = one_proj_sorted(
                        s[above_bound[-1]+1:], a_val - u_b * len(above_bound),
                        u_b=u_b)
    else:
        new_s = s - np.divide((np.cumsum(s) - a_val), np.arange(1, p+1))
        isneg = np.where(new_s < 0)[0]
        if len(isneg) > 0:
            first_neg = isneg[0]
            y = np.zeros(p)
            y[:first_neg] = s[:first_neg] + (
                a_val - np.sum(s[:first_neg])) / first_neg

    return(y[invperm])


def one_proj_sparse(s_vec, a_val, u_b=None, k_sparse=None):
    p = len(s_vec)
    # s_sum = np.sum(s_vec)
    perm = np.argsort(-s_vec)
    invperm = np.argsort(perm)
    s = s_vec[perm]
    if k_sparse is not None:
        if k_sparse < p:
            yy = np.zeros(p)
            yy[:k_sparse] = one_proj_sorted(s[:k_sparse], a_val,
                                            u_b=u_b)
            return(yy[invperm])

    return(one_proj_sorted(s_vec, a_val, u_b=u_b))


def proj2dupli(S, Z, A, u_b=None, k_sparse=None, include_main_diag=True):
    """
    projects the matrix S onto the set of matrices X such that
    Z X Z.T = A
    Z is n x N assignment matrix, with Z @ 1_N = C, and 1_n.T @ Z = 1_N
    """
    # Should check symmetry or triangularity and switch to lower triangular
    if include_main_diag:
        k = 0
    else:
        k = -1
    (N, N1) = S.shape
    assert(N == N1)
    (n, N2) = Z.shape
    assert(N2 == N)
    (n1, n2) = A.shape
    assert(n == n1 and n == n2)

    Atri = np.tril(A, k=k)
    (sis, sjs, svs) = find(Atri)
    n_vals = len(sis)

    (i_tril, j_tril) = np.tril_indices(N, k=k, m=N)
    ind_tril = np.ravel_multi_index((i_tril, j_tril), (N, N))
    tril_argsort = np.argsort(ind_tril)
    tril_map = np.zeros(N**2, dtype=int)
    tril_map[ind_tril] = tril_argsort
    s_flat = S.flatten()[ind_tril]
    s_new = np.zeros(len(ind_tril))

    these_ks = [np.where(Z[ii])[0] for ii in sis]
    these_ls = [np.where(Z[jj])[0] for jj in sjs]
    # Stay in lower triangle
    # the_ks = np.maximum(these_ks, these_ls)
    # the_ls = np.minimum(these_ks, these_ls)
    for k in range(n_vals):
        # s_vec = S[these_ks, these_ls].flatten()
        # if len(s_vec) == 1:
        #     pass
        # else:
        my_ks = these_ks[k]
        my_ls = these_ls[k]
        my_ks, my_ls = np.meshgrid(these_ks[k], these_ls[k])
        my_ks = my_ks.flatten()
        my_ls = my_ls.flatten()
        # Stay in lower triangle
        the_ks = np.maximum(my_ks, my_ls)
        the_ls = np.minimum(my_ks, my_ls)
        these_ind = np.ravel_multi_index((the_ks, the_ls), (N, N))
        these_ind = tril_map[these_ind]
        # if len(these_ind) == 1:
        if type(these_ind) is not np.ndarray:
            s_new[these_ind] = svs[k]
        else:
            s_vec = s_flat[these_ind]
            # Perform the projection for subscript pair (i,j)
            s_new[these_ind] = one_proj_sparse(s_vec,
                                               svs[k], u_b=None, k_sparse=None)

    xnew = np.zeros(N**2)
    xnew[ind_tril] = s_new
    X_proj = np.reshape(xnew, (N, N))
    if not include_main_diag:
        X_proj += X_proj.T
    else:
        X_proj += np.tril(X_proj, k=-1).T

    return(X_proj)


if __name__ == 'main':

    s_vec = np.array([1, 3, 12, 10])
    a_val = 40
    u_b = 12
    s_param = 3

    yy = one_proj_sorted(s_vec, a_val, u_b=u_b)

    yyy = one_proj_sparse(s_vec, a_val, u_b=u_b, k_sparse=s_param)

    # S2 = S.copy()
    # Noise = np.random.rand(N,N)
    # Noise += Noise.T
    # Noise *= 0.1 * S2.max()
    # S2 = S2 + Noise
    # plt.matshow(S2); plt.show()
    # Sproj = proj2dupli(S2, Z, A, u_b=None, k_sparse=None, include_main_diag=True)
    #
    # fig, axes = plt.subplots(1, 3)
    # axes[0].matshow(S)
    # axes[1].matshow(Sproj)
    # axes[2].matshow(S2)
    # plt.show()
