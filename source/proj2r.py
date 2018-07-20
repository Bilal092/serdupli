#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of Robinson matrices
"""
import sys
import numpy as np
from scipy.optimize import linprog
import mosek
from scipy.sparse import issparse, coo_matrix


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def proj2Rmat(X, do_strong=True, include_main_diag=True, verbose=0):
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
    # Find ``inverse'' of ind_tril
    tril_argsort = np.argsort(ind_tril)
    tril_map = np.zeros(n**2, dtype=int)
    tril_map[ind_tril] = tril_argsort
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
        assert(np.all(these_i >= these_j))
        these_ind = np.ravel_multi_index((these_i, these_j), (n, n))
        these_ind = tril_map[these_ind]
        alpha_idx = 2 * n_tri + k_diag - first_diag
        diag_len = len(these_ind)
        if k_diag > first_diag:
            # Diagonal entries lower than slack variable just above
            next_ineq_idx = ajs[-1] + 1
            ineq_idx = np.arange(next_ineq_idx, next_ineq_idx + diag_len)

            i_idx = [alpha_idx - 1] * diag_len  # might be a type issue
            i_idx = np.array(i_idx)
            sign_ = -np.ones(diag_len, dtype=int)
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
            i_idx = np.array(i_idx)
            sign_ = np.ones(diag_len)
            ais = np.append(ais, i_idx)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, sign_)

            ais = np.append(ais, these_ind)
            ajs = np.append(ajs, ineq_idx)
            avs = np.append(avs, -sign_)

            b_ub = np.append(b_ub, np.zeros(diag_len))

    # Build the inequality matrix
    A_ub = coo_matrix((avs, (ajs, ais)), dtype=int)
    (n_cons, n_var) = A_ub.shape
    # Build the vector c for the linear program
    c = np.zeros(n_var)
    c[n_tri:2*n_tri] = 1
    # Add non-negativity constraints ? (implemented by default)
    #
    # method = 'simplex'
    # # Call the solver
    # res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=method,
    #               options={"disp": True})

    inf = 10 * max(xval.max(), -xval.min())
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            if verbose > 0:
                # Attach a log stream printer to the task
                task.set_Stream(mosek.streamtype.log, streamprinter)
            # Bound key for constraints
            bkc = [mosek.boundkey.up] * n_cons
            # Bound values for constraints
            blc = [0] * n_cons
            buc = list(b_ub)
            # Bound keys for variables
            bkx = [mosek.boundkey.fr] * n_var
            # Bound values for Variables
            blx = [-inf] * n_var
            bux = [+inf] * n_var
            # Vector c
            c = list(c)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(n_cons)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(n_var)

            # Build constraint matrix
            task.putaijlist(ajs, ais, avs)
            # Add linear coeff
            task.putclist(list(range(n_var)), c)
            # Add variable bounds
            task.putvarboundlist(list(range(n_var)), bkx, blx, bux)
            # Add constraint bounds
            task.putconboundlist(list(range(n_cons)), bkc, blc, buc)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            # Call solver
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.bas)

            if (solsta == mosek.solsta.optimal or
                    solsta == mosek.solsta.near_optimal):
                xx = [0.] * n_var
                task.getxx(mosek.soltype.bas,  # Request the basic solution.
                           xx)
            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer or
                  solsta == mosek.solsta.near_dual_infeas_cer or
                  solsta == mosek.solsta.near_prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

    # xnewval = xx[:n_tri]
    xnew = np.zeros(n**2)
    xnew[ind_tril] = xx[:n_tri]
    X_proj = np.reshape(xnew, (n, n))
    if not include_main_diag:
        X_proj += X_proj.T
    else:
        X_proj += np.tril(X_proj, k=-1).T

    return X_proj


if __name__ == 'main':

    import matplotlib.pyplot as plt
    from mdso import SimilarityMatrix

    n = 200
    type_noise = 'gaussian'
    ampl_noise = 0.5
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    mat = data_gen.sim_matrix
    X = mat.copy()
    include_main_diag = True
    X_proj = proj2Rmat(mat)
    plt.matshow(X_proj)
    plt.show()
