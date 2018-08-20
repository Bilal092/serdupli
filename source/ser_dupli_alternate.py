#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import numpy as np
from scipy.sparse import find, coo_matrix, issparse
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from mdso import SpectralOrdering, SpectralBaseline

from proj2r import proj2Rmat
from proj2dupli import proj2dupli
from spectral_eta_trick_ import SpectralEtaTrick

from eval_dupli import eval_assignments

from gen_data import gen_chr_mat

from mdso.spectral_embedding_ import spectral_embedding


def is_symmetric(m):
    """Check if a sparse matrix is symmetric
    (from Saullo Giovani)

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')
    if issparse(m):
        if not isinstance(m, coo_matrix):
            m = coo_matrix(m)

        r, c, v = m.row, m.col, m.data
        tril_no_diag = r > c
        triu_no_diag = c > r

        if triu_no_diag.sum() != tril_no_diag.sum():
            return False

        rl = r[tril_no_diag]
        cl = c[tril_no_diag]
        vl = v[tril_no_diag]
        ru = r[triu_no_diag]
        cu = c[triu_no_diag]
        vu = v[triu_no_diag]

        sortl = np.lexsort((cl, rl))
        sortu = np.lexsort((ru, cu))
        vl = vl[sortl]
        vu = vu[sortu]

        check = np.allclose(vl, vu)

    else:
        check = np.allclose(m, m.T, atol=1e-6)

    return check


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
    # axes[1, 1].scatter(iiz, jjz, marker='.', s=4)
    # axes[1, 0].spy(Z)
    # if Z_true is not None:
    #     (izt, jzt, _) = find(Z_true)
    #     axes[1, 1].scatter(izt, jzt, marker='d', s=10, facecolors='none',
    #                        edgecolors='r')
    axes[1, 1].plot(perm, 'o', mfc='none')
    # axes[1, 1].plot(perm[:, 0], 'o', mfc='none')
    # axes[1, 1].scatter(perm[:, 0], perm[:, 1], c=np.arange(len(perm[:, 0])))
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
        my_solver = SpectralEtaTrick(n_iter=10)
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
        # R_t -= R_t.min()

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


def linearized_cluster(X, K, scale_rho=False, accept_qtile=50):

    (n1, n) = X.shape
    assert(n1 == n)

    # Compute crossing curves
    if K is not None:
        m = n // K
    rho = np.zeros(n)
    rho_p = np.zeros(n)
    rho_m = np.zeros(n)
    X = np.tril(X, -1) - np.tril(X, -m)
    # X_a = np.tril(X[:, ::-1], -1)
    # X_a = X_a - np.tril(X_a, -m)
    X_a = X[:, ::-1]
    X_p = X_a[:-1, :][:, :-1]

    for i in range(1, n-1):
        this_m = n - abs(n-1-2*i)
        # print(len(np.diag(X_a, n-1 - 2*i))-this_m)
        this_m = min(m, this_m)
        if this_m > 0:
            if not scale_rho:
                this_m = 1
            this_rho = (1./this_m) * np.trace(X_a, n-1 - 2*i)
        this_m = n - abs(n-1-2*i+1)
        this_m = min(m, this_m)
        if this_m > 0:
            if not scale_rho:
                this_m = 1
            this_rho += 1/2 * (1./this_m) * np.trace(X_a, n-1 - 2*i + 1)
        this_m = n - abs(n-1-2*i-1)
        this_m = min(m, this_m)
        if this_m > 0:
            if not scale_rho:
                this_m = 1
            this_rho += 1/2 * (1./this_m) * np.trace(X_a, n-1 - 2*i - 1)
        
        rho[i] = this_rho

    # smooth rho
    w_len = min(5, 1+n//20)
    rho_multi = np.zeros((n, 2 * w_len + 1))
    for k in range(1, w_len):
        rho_multi[:-k, k] = rho[k:]
        rho_multi[k:, k] += rho[:-k]
    rho_multi[:, 0] =  rho
    rho_avg = np.sum(rho_multi, axis=1)

    # Find valleys
    # We use a heuristic here...
    slope_sign = rho_avg[1:] - rho_avg[:-1]
    slope_sign = np.sign(slope_sign)
    slope_sign = np.append(slope_sign[0], slope_sign)
    sign_switch = slope_sign[1:] - slope_sign[:-1]
    bps = np.where(sign_switch)[0]
    ok_bps = np.zeros(1, dtype='int32')
    pctile = np.percentile(rho_avg, accept_qtile)
    for bp in bps:
        if np.all(slope_sign[bp-w_len:bp] == -1) and np.all(slope_sign[bp+1:bp+w_len] == 1):
            if rho[bp] < pctile:
                ok_bps = np.append(ok_bps, bp)
    ok_bps = np.append(ok_bps, n)
    return(ok_bps)


def clusterize_from_bps(X, bps, reord_clusters=True, reord_method=None):

    (N, N2) = X.shape
    assert(N == N2)
    n_clusters = len(bps) - 1

    if reord_clusters:
        permu = np.zeros(0, dtype='int32')
        if reord_method == 'eta-trick':
            my_method = SpectralEtaTrick(n_iter=10)
        elif reord_method == 'mdso':
            my_method = SpectralOrdering()
        else:
            my_method = SpectralBaseline()

    x_flat = X.flatten()
    s_clus = np.zeros(N**2)
    for k_ in range(n_clusters):
        in_clst = np.arange(bps[k_], bps[k_+1])
        if not in_clst.size:
            print("empty cluster!")
            continue
        iis = np.repeat(in_clst, len(in_clst))
        jjs = np.tile(in_clst, len(in_clst))
        sub_idx = np.ravel_multi_index((iis, jjs), (N, N))
        s_clus[sub_idx] = x_flat[sub_idx]  # Projection on block matrices

        if reord_clusters:
            sub_mat = X.copy()[in_clst, :]
            sub_mat = sub_mat.T[in_clst, :].T
            sub_perm = my_method.fit_transform(sub_mat - sub_mat.min())
            sub_cc = in_clst[sub_perm]
            permu = np.append(permu, sub_cc)

    S_clus = np.reshape(s_clus, (N, N))

    if reord_clusters:
        return(S_clus, permu)
    else:
        return(S_clus)


def simple_clusters(X, K, reord_clusters=True, reord_method='eta-trick'):
    bps = linearized_cluster(X, K)

    return(clusterize_from_bps(X, bps, reord_clusters=reord_clusters))




def clusterize_mat(X, n_clusters, reord_mat=False, reord_method='eta-trick'):
    # X2 = X.copy()
    # minX = X2.min()
    # X2 -= minX
    if reord_mat:
        if reord_method == 'eta-trick':
            my_method = SpectralEtaTrick(n_iter=10)
        elif reord_method == 'mdso':
            my_method = SpectralOrdering()
        else:
            my_method = SpectralBaseline()

    ebd = spectral_embedding(X - X.min(), norm_laplacian='random_walk', norm_adjacency=False)
    N = X.shape[0]
    if n_clusters == 1:
        if reord_mat:
            return(X, np.arange(N))
        else:
            return(X)
    else:
        fied_vec = ebd[:, 0]
        fied_diff = abs(fied_vec[1:] - fied_vec[:-1])
        bps = np.append(0, np.argsort(-fied_diff)[:n_clusters-1])
        bps = np.append(bps, N)
        bps = np.sort(bps)
        x_flat = X.flatten()
        s_clus = np.zeros(N**2)
        if reord_mat:
            permu = np.zeros(0, dtype='int32')
        for k_ in range(n_clusters):
            in_clst = np.arange(bps[k_], bps[k_+1])
            if not in_clst.size:
                print("empty cluster!")
                continue
            iis = np.repeat(in_clst, len(in_clst))
            jjs = np.tile(in_clst, len(in_clst))
            sub_idx = np.ravel_multi_index((iis, jjs), (N, N))
            s_clus[sub_idx] = x_flat[sub_idx]  # Projection on block matrices

            if reord_mat:
                sub_mat = X.copy()[in_clst, :]
                sub_mat = sub_mat.T[in_clst, :].T
                sub_perm = my_method.fit_transform(sub_mat - sub_mat.min())
                sub_cc = in_clst[sub_perm]
                permu = np.append(permu, sub_cc)

        S_clus = np.reshape(s_clus, (N, N))
        if reord_mat:
            return(S_clus, permu)
        else:
            return(S_clus)


def ser_dupli_alt_clust3(A, C, seriation_solver='eta-trick', n_iter=100,
                  n_clusters=1, do_strong=False,
                  include_main_diag=True, do_show=True, Z_true=None,
                  cluster_interval=10):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    if seriation_solver == 'mdso':
        my_solver = SpectralOrdering(norm_laplacian='random_walk')
    elif seriation_solver == 'eta-trick':
        my_solver = SpectralEtaTrick(n_iter=20)
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
        permu = my_solver.fit_transform(S_t - S_t.min())

        is_identity = (np.all(permu == np.arange(N)) or
                       np.all(permu == np.arange(N)[::-1]))
        # if is_identity:
        #     break

        # S_tp = S_t[permu, :][:, permu]
        S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu, :].T

        # if False:  #(it % cluster_interval == 0) and (it > 0):
        #     R_clus, p2 = clusterize_mat(S_tp, n_clusters, reord_mat=False)
        # else:
        #     R_clus = S_tp
        #     p2 = np.arange(N)

        # R_clus = R_clus[p2, :]
        # R_clus = R_clus.T[:, p2].T

        # permu = permu[p2]

        if (it % cluster_interval == 0) and (it > 0):
            # R_clus, p2 = clusterize_mat(S_tp, n_clusters, reord_mat=True)
            R_clus, p2 = simple_clusters(S_tp, n_clusters, reord_clusters=True)
            R_clus = R_clus[p2, :]
            R_clus = R_clus.T[p2, :].T
            # R_clus = clusterize_mat(S_tp, n_clusters, reord_mat=False)
            # p2 = np.arange(N)
        else:
            R_clus = S_tp
            # R_clus = simple_clusters(S_tp, n_clusters)
            p2 = np.arange(N)

        permu = permu[p2]

        R_t = proj2Rmat(R_clus, do_strong=do_strong,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)
        print(R_t.min())

        # R_clus = clusterize_mat(R_t, n_clusters, reord_mat=False)
        # R_clus = simple_clusters(R_t, n_clusters, reord_clusters=False)

        # R_t -= R_t.min()

        Z = Z[:, permu]

        perm_tot = perm_tot[permu]

        r_clus_sym = is_symmetric(R_clus)
        r_sym = is_symmetric(R_t)
        s_sym = is_symmetric(S_tp)
        print(r_clus_sym, r_sym, s_sym)

        if do_show:
            title = "iter {}".format(int(it))
            if Z_true is not None:
                mean_dist, _, is_inv = eval_assignments(Z, Z_true)
                title += " mean dist {}".format(mean_dist)
                if is_inv:
                    Z = Z[:, ::-1]
            visualize_mat(R_clus, S_tp, R_t, Z, permu, title, Z_true=Z_true)

        S_t = proj2dupli(R_t, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)

    return(S_t, Z, R_clus, S_tp)


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
        my_solver = SpectralEtaTrick(n_iter=10)
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
    # max_val = S_t.max()

    perm_tot = np.arange(N)

    # Iterate
    for it in range(n_iter):
        # S_old
        # S_t -= S_t.min()  # to make sure it is non-negative after linprog

        permu1 = my_solver.fit_transform(S_t)
        S_t = S_t[permu1, :]
        S_t = S_t.T[permu1, :].T

        # Cluster the similarity matrix
        if (it % 10 == 0) and (it > 9):
            labels_ = cluster_solver.fit_predict(R_t.max() - R_t)

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

                # (iis, jjs) = np.meshgrid(in_clst, in_clst)
                # iis = iis.flatten()
                # jjs = jjs.flatten()
                iis = np.repeat(in_clst, len(in_clst))
                jjs = np.tile(in_clst, len(in_clst))
                sub_idx = np.ravel_multi_index((iis, jjs), (N, N))
                #
                # (iord, jord) = np.meshgrid(sub_cc, sub_cc)
                # iord = iord.flatten()
                # jord = jord.flatten()
                # sub_ord = np.ravel_multi_index((iord, jord), (N, N))
                #
                s_clus[sub_idx] = s_flat[sub_idx]  # Projection on block matrices
                # S_clus[in_clst, :][:, in_clst] += sub_mat

            is_identity = (np.all(permu == np.arange(N)) or
                        np.all(permu == np.arange(N)[::-1]))
            # if is_identity:
            #     break

            alpha_ = 0.
            S_clus = (1 - alpha_) * np.reshape(s_clus, (N, N)) + alpha_ * S_t
            # S_clus = np.reshape(s_clus, (N, N))
            S_tp = S_clus.copy()[permu, :]
            # S_tp = S_t.copy()[permu, :]
            S_tp = S_tp.T[permu, :].T
            # S_tp = S_tp.T[permu, :].T

        else:
            permu = np.arange(N)
            S_tp = S_t

        permu = permu1[permu]

        R_t = proj2Rmat(S_tp, do_strong=do_strong,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)
        # R_t = S_tp

        Z = Z[:, permu]

        perm_tot = perm_tot[permu]

        if do_show:
            title = "iter {}".format(int(it))
            if Z_true is not None:
                mean_dist, _, is_inv = eval_assignments(Z, Z_true)
                title += " mean dist {}".format(mean_dist)
                # if is_inv:
                #     Z = Z[:, ::-1]
            visualize_mat(S_t, S_tp, R_t, Z, permu, title, Z_true=Z_true)

        S_t = proj2dupli(R_t, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)

    return(S_t, Z)


def get_k_necks(X, k):
    """ function to get clustering of a Robinson matrix into a block matrix """
    # X = X.copy()
    # X -= X.min()
    # X = np.triu(X, 1)
    # if not isinstance(X, coo_matrix):
    #     X = coo_matrix(X)
    (iis, jjs, vvs) = find(np.triu(X - X.min(), 1))
    qv = np.percentile(vvs, 50)
    iis = iis[vvs>qv]
    jjs = jjs[vvs>qv]
    vvs = vvs[vvs>qv]

    i_diff = iis[1:] - iis[:-1]
    i_diff = np.append(i_diff, 1)
    idxs = np.where(i_diff)[0]
    i_env = iis[idxs]
    j_env = jjs[idxs]

    (iis, jjs, vvs) = find(np.tril(X - X.min(), -1))
    qv = np.percentile(vvs, 50)
    iis = iis[vvs>qv]
    jjs = jjs[vvs>qv]
    vvs = vvs[vvs>qv]
    i_diff = iis[1:] - iis[:-1]
    i_diff = np.append(1, i_diff)
    idxs2 = np.where(i_diff)[0]
    i_env = np.append(i_env, jjs[idxs2])
    j_env = np.append(j_env, iis[idxs2])

    # Get breakpoints
    contour_mat = coo_matrix((np.ones(len(i_env)), (i_env, j_env)), shape=X.shape)
    i_c, j_c, _ = find(contour_mat)
    d2diag = abs(i_c - j_c)
    slope = d2diag[1:] - d2diag[:-1]




    d2diag = abs(i_env - j_env)
    k_necks = np.argsort(d2diag)[:k]
    i_necks = j_env[k_necks] - d2diag[k_necks]
    return(i_necks)


def ser_dupli_alt_clust2(A, C, seriation_solver='eta-trick', n_iter=100,
                        n_clusters=8, do_strong=False, include_main_diag=True,
                        do_show=True, Z_true=None):

    (n_, n1) = A.shape
    n2 = len(C)
    N = int(np.sum(C))
    assert(n_ == n1 and n_ == n2)

    if seriation_solver == 'mdso':
        my_solver = SpectralOrdering(norm_laplacian='random_walk')
    elif seriation_solver == 'eta-trick':
        my_solver = SpectralEtaTrick(n_iter=10)
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
    # max_val = S_t.max()

    perm_tot = np.arange(N)

    # Iterate
    for it in range(n_iter):
        # S_old
        # S_t -= S_t.min()  # to make sure it is non-negative after linprog

        # Reorder the matrix
        permu = my_solver.fit_transform(S_t)
        # S_tp = S_t[permu, :][:, permu]
        S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu, :].T

        R_t = proj2Rmat(S_tp, do_strong=do_strong,
                        include_main_diag=include_main_diag, verbose=0,
                        u_b=max_val)
        print(R_t.min())
        R_t -= R_t.min()
        # (iis, jjs, vvs) = find(R_t)
        # qv = np.percentile(vvs, 50)
        # iis = iis[vvs>qv]
        # jjs = jjs[vvs>qv]
        # vvs = vvs[vvs>qv]
        # R_t = coo_matrix((vvs, (iis, jjs)), shape=R_t.shape)
        # R_t = R_t.toarray()

        ebd = spectral_embedding(R_t, norm_laplacian=False)
        if n_clusters > 1:
            # fied_vec = ebd[:, 0]
            # fied_diff = abs(fied_vec[1:] - fied_vec[:-1])
            # bps = np.append(0, np.argsort(-fied_diff)[:n_clusters-1])
            # bps = np.append(bps, N)
            # bps = np.sort(bps)

            # bps = get_k_necks(R_t, n_clusters-1)
            # bps = np.append(0, bps)
            # bps = np.append(bps, N)
            # bps = np.sort(bps)
            bps = np.array([0, N])
        else:
            bps = np.array([0, N])
        print(bps)
        labels_ = np.zeros(N)
        # for labels_[bps[]]


        Z = Z[:, permu]

        # perm_tot = perm_tot[permu]

        # Cluster the similarity matrix
        # labels_ = cluster_solver.fit_predict(R_t.max() - R_t)
        # print(sum(labels_))

        # Reorder each cluster
        s_clus = np.zeros(N**2)  # TODO: adapt to the sparse case
        s_flat = R_t.flatten()
        permu2 = np.zeros(0, dtype='int32')
        # permu = np.arange(N)
        
        for k_ in range(n_clusters):
            # in_clst = np.where(labels_ == k_)[0]
            in_clst = np.arange(bps[k_], bps[k_+1])
            # sub_mat = R_t[in_clst, :]
            # sub_mat = sub_mat.T[in_clst, :].T
            # sub_perm = my_solver.fit_transform(sub_mat)
            # sub_cc = in_clst[sub_perm]
            sub_cc = in_clst

            # inv_sub_perm = np.argsort(sub_perm)
            # permu[in_clst] = sub_cc  # in_clst[inv_sub_perm]
            # permu[in_clst] = in_clst[inv_sub_perm]
            permu2 = np.append(permu2, sub_cc)

            # (iis, jjs) = np.meshgrid(in_clst, in_clst)
            # iis = iis.flatten()
            # jjs = jjs.flatten()
            iis = np.repeat(in_clst, len(in_clst))
            jjs = np.tile(in_clst, len(in_clst))
            sub_idx = np.ravel_multi_index((iis, jjs), (N, N))
            #
            # (iord, jord) = np.meshgrid(sub_cc, sub_cc)
            # iord = iord.flatten()
            # jord = jord.flatten()
            # sub_ord = np.ravel_multi_index((iord, jord), (N, N))
            #
            s_clus[sub_idx] = s_flat[sub_idx]  # Projection on block matrices
            # S_clus[in_clst, :][:, in_clst] += sub_mat

        # is_identity = (np.all(permu == np.arange(N)) or
        #                np.all(permu == np.arange(N)[::-1]))
        # if is_identity:
        #     break

        alpha_ = 0.
        S_clus = (1 - alpha_) * np.reshape(s_clus, (N, N)) + alpha_ * S_t
        # S_clus = np.reshape(s_clus, (N, N))
        S_tp = S_clus.copy()[permu2, :]
        # S_tp = S_t.copy()[permu, :]
        S_tp = S_tp.T[permu2, :].T
        # S_tp = S_tp.T[permu, :].T

        # R_t = proj2Rmat(S_tp, do_strong=do_strong,
        #                 include_main_diag=include_main_diag, verbose=0,
        #                 u_b=max_val)
        # R_t = S_tp

        double_perm = permu[permu2]
        Z = Z[:, permu2]

        perm_tot = perm_tot[double_perm]

        if do_show:
            title = "iter {}".format(int(it))
            if Z_true is not None:
                mean_dist, _, is_inv = eval_assignments(Z, Z_true)
                title += " mean dist {}".format(mean_dist)
                # if is_inv:
                #     Z = Z[:, ::-1]
            visualize_mat(S_t, S_tp, R_t, Z, ebd, title, Z_true=Z_true)

        S_t = proj2dupli(S_tp, Z, A, u_b=max_val, k_sparse=None,
                         include_main_diag=include_main_diag)

    return(S_t, Z, R_t)



# print(__name__)
if __name__ == '__main__':

    from mdso import SimilarityMatrix
    import matplotlib.pyplot as plt
    from scipy.linalg import toeplitz

    from gen_data import gen_dupl_mat

    n = 250
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
    S = S + 0.5 * gen_chr_mat(n, 8)
    #
    # n = 200
    # my_diag = np.zeros(n)
    # my_diag[:int(n//5)] = 1
    # S = toeplitz(my_diag)
    n_by_N = 0.65
    prop_dupli = 1
    rand_seed = 1
    np.random.seed(rand_seed)

    (Z, A, C) = gen_dupl_mat(S, n_by_N, prop_dupli=prop_dupli,
                             rand_seed=rand_seed)
    plt.matshow(S)
    plt.show()

    plt.matshow(A)
    plt.show()
    # plt.close()

    Z_true = Z.toarray()

    # ser_dupli_alt(A, C, seriation_solver='mdso', n_iter=100,
    #               do_strong=False,
    #               include_main_diag=True, do_show=True, Z_true=Z_true)

    (S_t, Z, R_clus, S_tp) = ser_dupli_alt_clust3(A, C, seriation_solver='eta-trick', n_iter=100,
                        n_clusters=3, do_strong=False, include_main_diag=True,
                        do_show=True, Z_true=Z_true, cluster_interval=2)

    # (S_, Z_, R_) = ser_dupli_alt_clust2(A, C, seriation_solver='eta-trick', n_iter=20,
    #                     n_clusters=1, do_strong=False, include_main_diag=True,
    #                     do_show=True, Z_true=Z_true)

    # import scipy.io
    # scipy.io.savemat(
    #     '/Users/antlaplante/THESE/SeriationDuplications/data/pythonvars.mat',
    #     mdict={'Z': Z_true, 'A': A, 'S': S, 'C': C})
