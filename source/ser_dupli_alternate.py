#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mdso import SpectralOrdering, SpectralBaseline

from proj2r import proj2Rmat
from proj2dupli import proj2dupli
from spectral_eta_trick_ import SpectralEtaTrick


def visualize_mat(S_t, S_tp, R_t, Z, perm, title, Z_true=None, fig_nb=1):

    # fig = plt.figure(fig_nb)
    # ax1 = plt.subplot2grid((5, 100), (0, 0), rowspan=4, colspan=100)
    # # ax2 = plt.subplot2grid((5,100),(4,19),colspan=62)
    # ax2 = plt.subplot2grid((5, 100), (0, 0), rowspan=4, colspan=100)
    # ax1.matshow(S_t)
    # ax2.matshow(R_t)
    # # ax2.axis('off')
    # # ax2.set_aspect(3)
    # # plt.title(str(len(ORDER)))
    # plt.draw()
    # plt.pause(0.01)
    plt.close()
    fig, axes = plt.subplots(2, 2)
    cax = axes[0, 0].matshow(S_t)
    fig.colorbar(cax)
    axes[0, 1].matshow(R_t)
    axes[1, 0].matshow(S_tp)
    # (iiz, jjz, _) = find(Z)
    # axes[1, 1].scatter(iiz, jjz)
    # # axes[1, 0].spy(Z)
    # if Z_true is not None:
    #     (izt, jzt, _) = find(Z_true)
    #     axes[1, 1].scatter(izt, jzt)
    axes[1, 1].plot(perm, 'o', mfc='none')
    # axes[2].matshow(S2)
    plt.title(title)
    plt.draw()
    plt.pause(0.01)

    return


def ser_dupli_alt(A, C, seriation_solver='eta-trick', n_iter=100,
                  include_main_diag=True, do_show=True, Z_true=None):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    if seriation_solver == 'mdso':
        my_solver = SpectralOrdering(norm_laplacian='unnormalized')
    elif seriation_solver == 'eta-trick':
        my_solver = SpectralEtaTrick()
    else:  # use basic spectral Algorithm from Atkins et. al.
        my_solver = SpectralBaseline()

    # Initialization
    Z = np.zeros((n_, N))
    jj = 0
    for ii in range(n_):  # TODO : make this faster ?
        Z[ii, jj:jj+C[ii]] = 1
        jj += C[ii]
    dc = np.diag(1./C)

    # Z = Z_true.copy()
    S_t = Z.T @ dc @ A @ dc @ Z

    max_val = A.max()

    # pp = np.random.permutation(N)
    # S_t = S[pp,:][:,pp]
    # Z = Z_true.copy()
    # Z = Z[:, pp]

    perm_tot = np.arange(N)

    # Iterate
    for it in range(n_iter):
        # S_old
        S_t -= S_t.min()  # to make sure it is non-negative after linprog
        permu = my_solver.fit_transform(S_t)

        # S_tp = S_t[permu, :][:, permu]
        S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu, :].T

        R_t = proj2Rmat(S_tp, do_strong=True,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)

        Z = Z[:, permu]

        perm_tot = perm_tot[permu]

        if do_show:
            title = "iter {}".format(int(it))
            visualize_mat(S_t, S_tp, R_t, Z, permu, title, Z_true=Z_true)

        S_t = proj2dupli(R_t, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)


    return(S_t, Z)


if __name__ == 'main':

    from mdso import SimilarityMatrix
    import matplotlib.pyplot as plt
    from scipy.linalg import toeplitz

    n = 150
    type_noise = 'gaussian'
    ampl_noise = 0.5
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    S = data_gen.sim_matrix
    #
    # n = 200
    # my_diag = np.zeros(n)
    # my_diag[:int(n//5)] = 1
    # S = toeplitz(my_diag)
    # n_by_N = 0.75
    # prop_dupli = 1
    # rand_seed = 1

    (Z, A, C) = gen_dupl_mat(S, n_by_N, prop_dupli=prop_dupli,
                             rand_seed=rand_seed)
    plt.matshow(S)
    plt.show()

    Z_true = Z.toarray()

    import scipy.io
    scipy.io.savemat(
        '/Users/antlaplante/THESE/SeriationDuplications/data/pythonvars.mat',
        mdict={'Z': Z_true, 'A': A, 'S': S, 'C': C})
