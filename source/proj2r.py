#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of Robinson matrices
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import issparse, coo_matrix


def proj2Rmat(X, do_strong=True, include_main_diag=True):
    """
    Projects a matrix to the set of Robinson matrices.
    Performs \min_{A} \sum_{ij} |A_{ij} - X_{ij}| such that A is Robinson,
    through a linear program min_{xi, a, alpha} \sum xi_{ij}, such that \...
    xi_{ij} >= X_{ij} - A_{ij}
    xi_{ij} >= A_{ij} - X_{ij}
    A_{ij} >= alpha(|i-j|)
    A_{ij} <= alpha(|i-j|-1)

    The variable used in linprog is the concatenation of (a, xi, alpha)

    Parameters
    ----------
    X : numpy array or scipy.sparse array
        input matrix to project on the set of R matrices

    do_strong : bool, default True
        whether to project on the set of strong R matrices or standard R
        matrices

    """
    (n, n2) = X.shape
    assert(n == n2)
    # TODO: check X is symmetric or triangular, and switch to lower triangular
    if include_main_diag:
        n_tri = n * (n+1) // 2
        k = 0
    else:
        n_tri = n * (n-1) // 2
        k = -1
    (i_tril, j_tril) = np.tril_indices(n, k=k, m=n)
    ind_tril = np.ravel_multi_index((i_tril, j_tril), (n, n))
    xval = X.flatten()[ind_tril]
    ais = []
    ajs = []
    avs = []
    b_ub = np.zeros(0)
    # Add absolute-value related inequalities to A_ub
    # First direction of absolute value
    ineq_idx = np.arange(n_tri)

    i_idx = np.arange(n_tri)
    sign_ = -np.ones(n_tri, dtype=int)
    ais.append(i_idx)
    ajs.append(ineq_idx)
    avs.append(sign_)

    i_idx = np.arange(n_tri, 2 * n_tri)
    ais.append(i_idx)
    ajs.append(ineq_idx)
    avs.append(sign_)

    b_ub = np.append(b_ub, -xval)

    # Second direction of absolute value
    ineq_idx = np.arange(n_tri, 2 * n_tri)

    i_idx = np.arange(n_tri)
    sign_ = np.ones(n_tri, dtype=int)
    ais.append(i_idx)
    ajs.append(ineq_idx)
    avs.append(sign_)

    i_idx = np.arange(n_tri, 2 * n_tri)
    sign_ *= -1
    ais.append(i_idx)
    ajs.append(ineq_idx)
    avs.append(sign_)

    b_ub = np.append(b_ub, xval)

    # Add R-constraints related inequalities to A_ub
    first_diag = 0 if include_main_diag else 1
    (i_diag_sub, j_diag_sub) = np.diag_indices(n)
    for k_diag in range(first_diag, n):
        these_i = i_diag_sub[k_diag:]
        these_j = j_diag_sub[:-k_diag]
        these_ind = np.ravel_multi_index((these_i, these_j), (n, n))
        alpha_idx = 2 * n_tri + k_diag - first_diag
        diag_len = len(these_ind)
        if k_diag > first_diag:
            # Diagonal entries lower than slack variable just above
            next_ineq_idx = ajs[-1] + 1
            ineq_idx = np.arange(next_ineq_idx, next_ineq_idx + diag_len)

            i_idx = [alpha_idx - 1] * diag_len  # might be a type issue
            sign_ = np.ones(diag_len)
            ais.append(i_idx)
            ajs.append(ineq_idx)
            avs.append(sign_)

            ais.append(these_ind)
            ajs.append(ineq_idx)
            avs.append(-sign_)

            b_ub = np.append(b_ub, np.zeros(diag_len))

        if k_diag < n-1:
            # Diagonal entries larger than slack variable just below
            next_ineq_idx = ajs[-1] + 1
            ineq_idx = np.arange(next_ineq_idx, next_ineq_idx + diag_len)

            i_idx = [alpha_idx] * diag_len  # might be a type issue
            sign_ = -np.ones(diag_len)
            ais.append(i_idx)
            ajs.append(ineq_idx)
            avs.append(sign_)

            ais.append(these_ind)
            ajs.append(ineq_idx)
            avs.append(-sign_)

            b_ub = np.append(b_ub, np.zeros(diag_len))

    # Build the inequality matrix
    A_ub = coo_matrix((avs, (ais, ajs)), dtype=int)
    c = 
