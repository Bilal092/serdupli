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
    ais = np.zeros(0, dtype=int)
    ajs = np.zeros(0, dtype=int)
    avs = np.zeros(0, dtype=int)
    b_ub = np.zeros(0)
    # Add absolute-value related inequalities to A_ub
    # First direction of absolute value
    ineq_idx = np.arange(n_tri)

    i_idx = np.arange(n_tri)
    sign_ = -np.ones(n_tri, dtype=int)
    ais = np.append(ais, i_idx)
    ajs = np.append(ajs, ineq_idx)
    avs = np.append(avs, sign_)

    i_idx = np.arange(n_tri, 2 * n_tri)
    ais = np.append(ais, i_idx)
    ajs = np.append(ajs, ineq_idx)
    avs = np.append(avs, sign_)

    b_ub = np.append(b_ub, -xval)

    # Second direction of absolute value
    ineq_idx = np.arange(n_tri, 2 * n_tri)

    i_idx = np.arange(n_tri)
    sign_ = np.ones(n_tri, dtype=int)
    ais = np.append(ais, i_idx)
    ajs = np.append(ajs, ineq_idx)
    avs = np.append(avs, sign_)

    i_idx = np.arange(n_tri, 2 * n_tri)
    sign_ *= -1
    ais = np.append(ais, i_idx)
    ajs = np.append(ajs, ineq_idx)
    avs = np.append(avs, sign_)

    b_ub = np.append(b_ub, xval)

    # Add R-constraints related inequalities to A_ub
    first_diag = 0 if include_main_diag else 1
    (i_diag_sub, j_diag_sub) = np.diag_indices(n)
    for k_diag in range(first_diag, n):
        if k_diag > 0:
            these_i = i_diag_sub[k_diag:]
            these_j = j_diag_sub[:-k_diag]
        else:
            these_i = i_diag_sub[:]
            these_j = j_diag_sub[:]
        these_ind = np.ravel_multi_index((these_i, these_j), (n, n))
        alpha_idx = 2 * n_tri + k_diag - first_diag
        diag_len = len(these_ind)
        if k_diag > first_diag:
            # Diagonal entries lower than slack variable just above
            next_ineq_idx = ajs[-1] + 1
            ineq_idx = np.arange(next_ineq_idx, next_ineq_idx + diag_len)

            i_idx = [alpha_idx - 1] * diag_len  # might be a type issue
            sign_ = np.ones(diag_len)
            ais = np.append(ais, i_idx)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, sign_)

            ais = np.append(ais, these_ind)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, -sign_)

            b_ub = np.append(b_ub, np.zeros(diag_len))

        if k_diag < n-1:
            # Diagonal entries larger than slack variable just below
            next_ineq_idx = ajs[-1] + 1
            ineq_idx = np.arange(next_ineq_idx, next_ineq_idx + diag_len)

            i_idx = [alpha_idx] * diag_len  # might be a type issue
            sign_ = -np.ones(diag_len)
            ais = np.append(ais, i_idx)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, sign_)

            ais = np.append(ais, these_ind)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, -sign_)

            b_ub = np.append(b_ub, np.zeros(diag_len))

    # Build the inequality matrix
    A_ub = coo_matrix((avs, (ajs, ais))), dtype=int)
    (n_cons, n_var) = A_ub.shape
    # Build the vector c for the linear program
    c = np.zeros(n_var)
    c[n_tri:2*n_tri] = 1
    # Add non-negativity constraints ? (implemented by default)

    method = 'simplex'
    # Call the solver
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=method,
                  options={"disp": True})
    return res


if __name__ == 'main':
    from mdso import SimilarityMatrix

    n = 500
    type_noise = 'gaussian'
    ampl_noise = 0.5
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    mat = data_gen.sim_matrix
    res = proj2Rmat(mat)
