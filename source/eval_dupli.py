#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess the results of Seriation with duplication experiments
"""
import numpy as np


def eval_twins(z_1, z_2, weighted=False):
    """
    Computes the sum of the distances between non-zero indices of z_1 and z_2.
    If z_1 and z_2 have same length, the optimal matching is equivalent to
    sorting z_1 and z_2.
    """

    idx_1 = np.nonzero(z_1)
    idx_2 = np.nonzero(z_2)

    n_1 = len(idx_1)
    n_2 = len(idx_2)
    if n_1 != n_2:
        raise ValueError("z_1 and z_2 must have same number of non-zeros")

    idx_1 = np.sort(idx_1)
    idx_2 = np.sort(idx_2)

    dif = abs(idx_1 - idx_2)

    dist = np.sum(dif)

    if weighted:
        dist /= n_1

    return(dist)


def eval_assignments(Z, Z_true, return_summary=True, return_inv=True):
    """ Evaluate score between two assignment matrices in the Seriation with
    Duplications problem.
    """

    (n, N) = Z.shape
    (n_, N_) = Z_true.shape
    if n > N:
        Z = Z.T
        nn = n
        n = N
        N = nn
    if n_ > N_:
        Z_true = Z_true.T
        nn_ = n_
        n_ = N_
        N_ = nn_
    if not(n == n_ and N == N_):
        raise ValueError("Z and Z_true must have the same shape"
                         " ({} vs {})".format(Z.shape, Z_true.shape))

    (ii_1, jj_1) = np.nonzero(Z)
    (ii_2, jj_2) = np.nonzero(Z_true)
    (_, jj_inv) = np.nonzero(Z[:, ::-1])

    all_dist = abs(jj_1 - jj_2)
    inv_dist = abs(jj_inv - jj_2)

    if all_dist.sum() > inv_dist.sum():
        all_dist = inv_dist
        Z = Z[:, ::-1]
        is_inv = True
    else:
        is_inv = False

    C = Z @ np.ones(N)
    C = C.astype(int, copy=False)

    Zs = np.zeros((n, N))
    jj = 0
    for ii in range(n):
        Zs[ii, jj:jj+C[ii]] = 1
        jj += C[ii]

    dists = Zs @ all_dist
    dists = np.divide(dists, C)

    if return_summary:
        if return_inv:
            return(dists.mean(), dists.std(), is_inv)
        else:
            return(dists.mean(), dists.std())
    else:
        if return_inv:
            return(dists, is_inv)
        else:
            return(dists)
