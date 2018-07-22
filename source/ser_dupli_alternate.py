#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import sys
import numpy as np
from scipy.sparse import find
from mdso import SpectralOrdering, SpectralBaseline
import matplotlib.pyplot as plt

from proj2r import proj2Rmat
from proj2dupli import proj2dupli


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
    # axes[1, 0].spy(Z)
    # if Z_true is not None:
    #     axes[1, 0].spy(Z_true)
    axes[1, 1].plot(perm, 'o', mfc='none')
    # axes[2].matshow(S2)
    plt.title(title)
    plt.draw()
    plt.pause(0.01)

    return


def ser_dupli_alt(A, C, seriation_solver='mdso', n_iter=100,
                  include_main_diag=True, do_show=True, Z_true=None):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    # Initialization
    Z = np.zeros((n_, N))
    jj = 0
    for ii in range(n_):  # TODO : make this faster ?
        Z[ii, jj:jj+C[ii]] = 1
        jj += C[ii]
    dc = np.diag(1./C)

    Z = Z_true
    S_t = Z.T @ dc @ A @ dc @ Z

    perm_tot = np.arange(N)

    if seriation_solver == 'mdso':
        # my_solver = SpectralOrdering()
        my_solver = SpectralBaseline()
    # Iterate
    for it in range(n_iter):
        # S_old
        permu = my_solver.fit_transform(S_t)

        S_tp = S_t[permu, :][:, permu]

        R_t = proj2Rmat(S_tp, do_strong=True,
                        include_main_diag=include_main_diag, verbose=0)

        Z = Z[:, permu]

        S_t = proj2dupli(R_t, Z, A, u_b=None, k_sparse=None,
                         include_main_diag=include_main_diag)

        perm_tot = perm_tot[permu]

        if do_show:
            title = "iter {}".format(int(it))
            visualize_mat(S_t, S_tp, R_t, Z, perm_tot, title, Z_true=Z_true)

    return(S_t, Z)
