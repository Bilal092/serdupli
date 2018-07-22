#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempts to minimize \sum_{ij} A_{ij} |pi_i - pi_j| with eta-trick and spectral
ordering
"""
import warnings
import sys
import numpy as np
from scipy.sparse import issparse, coo_matrix, lil_matrix
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
            Xl = X.tolil(copy=True)
        else:
            Xl = X.copy()
        Xl = Xl[permut, :]
        Xl = Xl.T[permut, :].T

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    cax = ax.matshow(Xl, interpolation='nearest')
    # fig.colorbar(cax)
    plt.title(title)
    plt.draw()
    plt.pause(0.01)

    return


def spectral_eta_trick(X, n_iter=50, dh=1, p=1, return_score=False, do_plot=False):
    """
    Performs Spectral Eta-trick Algorithm from
    https://arxiv.org/pdf/1806.00664.pdf
    which calls several instances of the Spectral Ordering baseline (Atkins) to
    try to minimize 1-SUM or Huber-SUM (instead of 2-SUM)
    with the so-called eta-trick.
    """

    (n, n2) = X.shape
    assert(n == n2)

    spectral_algo = SpectralBaseline()

    best_perm = np.arange(n)
    best_score = n**(p+2)

    if issparse(X):
        if not isinstance(X, coo_matrix):
            X = coo_matrix(X)

        r, c, v = X.row, X.col, X.data
        eta_vec = np.ones(len(v))

        for it in range(n_iter):

            X_w = X.copy()
            X_w.data /= eta_vec

            new_perm = spectral_algo.fit_transform(X_w)
            if np.all(new_perm == best_perm):
                break

            new_score = p_sum_score(X, permut=new_perm, p=p)
            if new_score < best_score:
                best_perm = new_perm

            p_inv = np.argsort(new_perm)

            eta_vec = np.maximum(dh, abs(p_inv[r] - p_inv[c]))

            if do_plot:
                title = 'it {}, {}-SUM: {}'.format(it, p, new_score)
                plot_mat(X, permut=new_perm, title=title)

    else:
        eta_mat = np.ones((n, n))

        for it in range(n_iter):

            X_w = np.divide(X, eta_mat)
            new_perm = spectral_algo.fit_transform(X_w)
            if np.all(new_perm == best_perm):
                break

            new_score = p_sum_score(X, permut=new_perm, p=p)
            if new_score < best_score:
                best_perm = new_perm

            p_inv = np.argsort(new_perm)

            eta_mat = abs(np.tile(p_inv, n) - np.repeat(p_inv, n))
            eta_mat = np.reshape(eta_mat, (n, n))
            eta_mat = np.maximum(dh, eta_mat)

            if do_plot:
                title = 'it {}, {}-SUM: {}'.format(it, p, new_score)
                plot_mat(X, permut=new_perm, title=title)

    if return_score:
        return(best_perm, best_score)
    else:
        return(best_perm)
