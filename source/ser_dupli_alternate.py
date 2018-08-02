#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import numpy as np
from scipy.sparse import find
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from mdso import SpectralOrdering, SpectralBaseline

from source.proj2r import proj2Rmat
from source.proj2dupli import proj2dupli
from source.spectral_eta_trick_ import SpectralEtaTrick

from source.eval_dupli import eval_assignments


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
    fig.colorbar(cax, ax=axes[0, 0])
    cax2 = axes[0, 1].matshow(R_t)
    fig.colorbar(cax2, ax=axes[0, 1])
    cax3 = axes[1, 0].matshow(S_tp)
    fig.colorbar(cax3, ax=axes[1, 0])
    (iiz, jjz, _) = find(Z)
    axes[1, 1].scatter(iiz, jjz, marker='.', s=4)
    # axes[1, 0].spy(Z)
    if Z_true is not None:
        (izt, jzt, _) = find(Z_true)
        axes[1, 1].scatter(izt, jzt, marker='d', s=10, facecolors='none',
                           edgecolors='r')
    # axes[1, 1].plot(perm, 'o', mfc='none')
    # axes[2].matshow(S2)
    plt.title(title)
    plt.draw()
    plt.pause(0.01)

    return


def ser_dupli_alt(A, C, seriation_solver='eta-trick', n_iter=100,
                  do_strong=False,
                  include_main_diag=True, do_show=True, Z_true=None):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    if seriation_solver == 'mdso':
        my_solver = SpectralOrdering(norm_laplacian='random_walk')
    elif seriation_solver == 'eta-trick':
        my_solver = SpectralEtaTrick(n_iter=100)
    else:  # use basic spectral Algorithm from Atkins et. al.
        my_solver = SpectralBaseline()

    # Initialization
    Z = np.zeros((n_, N))
    jj = 0
    for ii in range(n_):  # TODO : make this faster ?
        Z[ii, jj:jj+C[ii]] = 1
        jj += C[ii]
    dc = np.diag(1./C)

    S_t = Z.T @ dc @ A @ dc @ Z

    max_val = A.max()

    perm_tot = np.arange(N)

    # Iterate
    for it in range(n_iter):
        # S_old
        # S_t -= S_t.min()  # to make sure it is non-negative after linprog
        # print(S_t.min())
        permu = my_solver.fit_transform(S_t)

        is_identity = (np.all(permu == np.arange(N)) or
                       np.all(permu == np.arange(N)[::-1]))
        # if is_identity:
        #     break

        # S_tp = S_t[permu, :][:, permu]
        S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu, :].T

        R_t = proj2Rmat(S_tp, do_strong=do_strong,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)
        print(R_t.min())
        R_t -= R_t.min()

        Z = Z[:, permu]

        perm_tot = perm_tot[permu]

        if do_show:
            title = "iter {}".format(int(it))
            if Z_true is not None:
                mean_dist, _, is_inv = eval_assignments(Z, Z_true)
                title += " mean dist {}".format(mean_dist)
                if is_inv:
                    Z = Z[:, ::-1]
            visualize_mat(S_t, S_tp, R_t, Z, permu, title, Z_true=Z_true)

        S_t = proj2dupli(R_t, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)

    return(S_t, Z)


def ser_dupli_alt_clust(A, C, seriation_solver='eta-trick', n_iter=100,
                        n_clusters=8, do_strong=False, include_main_diag=True,
                        do_show=True, Z_true=None):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    if seriation_solver == 'mdso':
        my_solver = SpectralOrdering(norm_laplacian='random_walk')
    elif seriation_solver == 'eta-trick':
        my_solver = SpectralEtaTrick(n_iter=100)
    else:  # use basic spectral Algorithm from Atkins et. al.
        my_solver = SpectralBaseline()

    cluster_solver = SpectralClustering(n_clusters=n_clusters,
                                        affinity='precomputed')

    # Initialization
    Z = np.zeros((n_, N))
    jj = 0
    for ii in range(n_):  # TODO : make this faster ?
        Z[ii, jj:jj+C[ii]] = 1
        jj += C[ii]
    dc = np.diag(1./C)

    S_t = Z.T @ dc @ A @ dc @ Z

    max_val = A.max()
    max_val = S_t.max()

    perm_tot = np.arange(N)

    # Iterate
    for it in range(n_iter):
        # S_old
        # S_t -= S_t.min()  # to make sure it is non-negative after linprog

        # Cluster the similarity matrix
        labels_ = cluster_solver.fit_predict(S_t.max() - S_t)

        # Reorder each cluster
        s_clus = np.zeros(N**2)  # TODO: adapt to the sparse case
        s_flat = S_t.flatten()
        permu = np.zeros(0, dtype='int32')
        # permu = np.arange(N)
        for k_ in range(n_clusters):
            in_clst = np.where(labels_ == k_)[0]
            sub_mat = S_t[in_clst, :]
            sub_mat = sub_mat.T[in_clst, :].T
            sub_perm = my_solver.fit_transform(sub_mat)
            sub_cc = in_clst[sub_perm]

            # inv_sub_perm = np.argsort(sub_perm)
            # permu[in_clst] = sub_cc  # in_clst[inv_sub_perm]
            # permu[in_clst] = in_clst[inv_sub_perm]
            permu = np.append(permu, sub_cc)

            (iis, jjs) = np.meshgrid(in_clst, in_clst)
            iis = iis.flatten()
            jjs = jjs.flatten()
            sub_idx = np.ravel_multi_index((iis, jjs), (N, N))
            #
            # (iord, jord) = np.meshgrid(sub_cc, sub_cc)
            # iord = iord.flatten()
            # jord = jord.flatten()
            # sub_ord = np.ravel_multi_index((iord, jord), (N, N))
            #
            s_clus[sub_idx] = s_flat[sub_idx]
            # S_clus[in_clst, :][:, in_clst] += sub_mat

        is_identity = (np.all(permu == np.arange(N)) or
                       np.all(permu == np.arange(N)[::-1]))
        # if is_identity:
        #     break

        alpha_ = 0.1
        S_clus = (1 - alpha_) * np.reshape(s_clus, (N, N)) + alpha_ * S_t
        # S_clus = np.reshape(s_clus, (N, N))
        S_tp = S_clus[permu, :]
        # S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu, :].T
        # S_tp = S_tp.T[permu, :].T

        R_t = proj2Rmat(S_tp, do_strong=do_strong,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)

        Z = Z[:, permu]

        perm_tot = perm_tot[permu]

        if do_show:
            title = "iter {}".format(int(it))
            if Z_true is not None:
                mean_dist, _, is_inv = eval_assignments(Z, Z_true)
                title += " mean dist {}".format(mean_dist)
                if is_inv:
                    Z = Z[:, ::-1]
            visualize_mat(S_t, S_tp, R_t, Z, permu, title, Z_true=Z_true)

        S_t = proj2dupli(R_t, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)

    return(S_t, Z)


if __name__ == 'main':

    from mdso import SimilarityMatrix
    import matplotlib.pyplot as plt
    from scipy.linalg import toeplitz

    from source.gen_data import gen_dupl_mat

    n = 150
    type_noise = 'gaussian'
    ampl_noise = 0.2
    type_similarity = 'LinearStrongDecrease'
    apply_perm = False
    # Build data matrix
    data_gen = SimilarityMatrix()
    data_gen.gen_matrix(n, type_matrix=type_similarity, apply_perm=apply_perm,
                        noise_ampl=ampl_noise, law=type_noise)
    S = data_gen.sim_matrix

    # S = gen_chr_mat(n, 3, type_mat=S)
    S = S + 1 * gen_chr_mat(n, 3)
    #
    # n = 200
    # my_diag = np.zeros(n)
    # my_diag[:int(n//5)] = 1
    # S = toeplitz(my_diag)
    n_by_N = 0.5
    prop_dupli = 1
    rand_seed = 1

    (Z, A, C) = gen_dupl_mat(S, n_by_N, prop_dupli=prop_dupli,
                             rand_seed=rand_seed)
    plt.matshow(S)
    plt.show()

    Z_true = Z.toarray()

    ser_dupli_alt(A, C, seriation_solver='eta-trick', n_iter=100,
                  do_strong=False,
                  include_main_diag=True, do_show=True, Z_true=Z_true)

    ser_dupli_alt_clust(A, C, seriation_solver='eta-trick', n_iter=100,
                        n_clusters=2, do_strong=False, include_main_diag=True,
                        do_show=True, Z_true=Z_true)

    # import scipy.io
    # scipy.io.savemat(
    #     '/Users/antlaplante/THESE/SeriationDuplications/data/pythonvars.mat',
    #     mdict={'Z': Z_true, 'A': A, 'S': S, 'C': C})
