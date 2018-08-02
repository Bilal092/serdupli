#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempts to minimize \sum_{ij} A_{ij} |pi_i - pi_j| with eta-trick and spectral
ordering
"""
import warnings
import sys
import numpy as np
from scipy.sparse import issparse, coo_matrix, lil_matrix, find
from scipy.linalg import toeplitz
from mdso import SpectralBaseline
import matplotlib.pyplot as plt


def p_sum_score(X, p=1, permut=None, normalize=False):
    """ computes the p-sum score of X or X[permut, :][:, permut] if permutation
    provided
    """
    if issparse(X):
        if not isinstance(X, coo_matrix):
            X = coo_matrix(X)

        r, c, v = X.row, X.col, X.data

        if permut is not None:
            d2diag = abs(permut[r] - permut[c])
        else:
            d2diag = abs(r - c)

        if p != 1:
            d2diag **= p

        prod = np.multiply(v, d2diag)
        score = np.sum(prod)

    else:
        if permut is not None:
            X_p = X.copy()[permut, :]
            X_p = X_p.T[permut, :].T
        else:
            X_p = X

        n = X_p.shape[0]
        d2diagv = np.arange(n)
        if p != 1:
            d2diagv **= p
        D2diag_mat = toeplitz(d2diagv)
        prod = np.multiply(X_p, D2diag_mat)
        score = np.sum(prod)

    return score


def plot_mat(X, title='', permut=None):

    if permut is not None:
        if issparse(X):
            (iis, jjs, _) = find(X)
            pis = permut[iis]
            pjs = permut[jjs]
            # Xl = X.tolil(copy=True)
        else:
            Xl = X.copy()
            Xl = Xl[permut, :]
            Xl = Xl.T[permut, :].T

    fig = plt.figure(1)
    plt.gcf().clear()
    axes = fig.subplots(1, 2)
    # ax = fig.add_subplot(111)
    if issparse(X):
        if permut is None:
            (pis, pjs, _) = find(X)
        # cax = ax.plot(pis, pjs, 'o', mfc='none')
        axes[0].plot(pis, pjs, 'o', mfc='none')
    else:
        # cax = ax.matshow(Xl, interpolation='nearest')
        axes[0].matshow(Xl, interpolation='nearest')
    if permut is not None:
        axes[1].plot(np.arange(len(permut)), permut, 'o', mfc='none')
    # fig.colorbar(cax)
    plt.title(title)
    plt.draw()
    plt.pause(0.01)

    return


def spectral_eta_trick(X, n_iter=50, dh=1, p=1, return_score=False,
                       do_plot=False, circular=False, norm_laplacian=None,
                       norm_adjacency=None, eigen_solver=None,
                       scale_embedding=False,
                       add_momentum=None):
    """
    Performs Spectral Eta-trick Algorithm from
    https://arxiv.org/pdf/1806.00664.pdf
    which calls several instances of the Spectral Ordering baseline (Atkins) to
    try to minimize 1-SUM or Huber-SUM (instead of 2-SUM)
    with the so-called eta-trick.
    """

    (n, n2) = X.shape
    assert(n == n2)

    spectral_algo = SpectralBaseline(circular=circular,
                                     norm_laplacian=norm_laplacian,
                                     norm_adjacency=norm_adjacency,
                                     eigen_solver=eigen_solver,
                                     scale_embedding=scale_embedding)

    best_perm = np.arange(n)
    best_score = n**(p+2)

    if issparse(X):
        if not isinstance(X, coo_matrix):
            X = coo_matrix(X)

        r, c, v = X.row, X.col, X.data
        eta_vec = np.ones(len(v))
        if add_momentum:
            eta_old = np.ones(len(v))

        for it in range(n_iter):

            X_w = X.copy()
            X_w.data /= eta_vec

            new_perm = spectral_algo.fit_transform(X_w)
            if np.all(new_perm == best_perm):
                break
            if new_perm[0] > new_perm[-1]:
                new_perm *= -1
                new_perm += (n)

            new_score = p_sum_score(X, permut=new_perm, p=p)
            if new_score < best_score:
                best_perm = new_perm

            p_inv = np.argsort(new_perm)

            eta_vec = abs(p_inv[r] - p_inv[c])
            if circular:
                # pass
                eta_vec = np.minimum(eta_vec, n - eta_vec)
            eta_vec = np.maximum(dh, eta_vec)

            if do_plot:
                title = "it %d, %d-SUM: %1.5e" % (it, p, new_score)
                plot_mat(X, permut=new_perm, title=title)

    else:
        eta_mat = np.ones((n, n))

        for it in range(n_iter):

            X_w = np.divide(X, eta_mat)
            new_perm = spectral_algo.fit_transform(X_w)
            if new_perm[0] > new_perm[-1]:
                new_perm *= -1
                new_perm += (n+1)
            if np.all(new_perm == best_perm):
                break

            new_score = p_sum_score(X, permut=new_perm, p=p)
            if new_score < best_score:
                best_perm = new_perm

            p_inv = np.argsort(new_perm)

            eta_mat = abs(np.tile(p_inv, n) - np.repeat(p_inv, n))
            if circular:
                # pass
                eta_mat = np.minimum(eta_mat, n - eta_mat)
            eta_mat = np.reshape(eta_mat, (n, n))
            eta_mat = np.maximum(dh, eta_mat)

            if do_plot:
                title = "it %d, %d-SUM: %1.5e" % (it, p, new_score)
                plot_mat(X, permut=new_perm, title=title)

    if return_score:
        return(best_perm, best_score)
    else:
        return(best_perm)


class SpectralEtaTrick():

    def __init__(self, n_iter=20, dh=1, return_score=False, circular=False,
                 norm_adjacency=None, eigen_solver=None):
        self.n_iter = n_iter
        self.dh = dh
        self.return_score = return_score
        self.circular = circular
        self.norm_adjacency = norm_adjacency
        self.eigen_solver = eigen_solver

    def fit(self, X):

        ordering_ = spectral_eta_trick(X, n_iter=self.n_iter, dh=self.dh,
                                       return_score=self.return_score,
                                       circular=self.circular,
                                       norm_adjacency=self.norm_adjacency,
                                       eigen_solver=self.eigen_solver)
        self.ordering = ordering_

        return self

    def fit_transform(self, X):

        self.fit(X)
        return self.ordering
