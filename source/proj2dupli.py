#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projects a matrix to the set of duplication constraints Z.T S Z = A
"""
import warnings
import sys
import numpy as np


def one_proj_sorted(s_vec, a_val, u_b=None):
    """
    Finds x = argmin |x - s_vec|, such that x_i >= 0 and sum_i x_i = a.
    """
    p = len(s_vec)
    s_sum = np.sum(s_vec)
    perm = np.argsort(-s_vec)
    invperm = np.argsort(perm)
    s = s_vec[perm]

    y = s + (a_val - s_sum) / p

    if a_val > s_sum:
        # y = s + (a_val - s_sum) / p
        # y += (a_val - s_sum) / p
        if u_b is not None:
            if y[0] > u_b:
                above_bound = np.where(y > u_b)[0]
                if len(above_bound) == p:
                    warnings.warn("pb. infeasible with upper bound {}".format(
                        u_b
                    ))
                    y[:] = u_b
                else:
                    y[above_bound] = u_b
                    y[above_bound[-1]+1:] = one_proj_sorted(
                        s[above_bound[-1]+1:], a_val - u_b * len(above_bound),
                        u_b=u_b)
    else:
        new_s = s - np.divide((np.cumsum(s) - a_val), np.arange(1, p+1))
        isneg = np.where(new_s < 0)[0]
        if len(isneg) > 0:
            first_neg = isneg[0]
            y = np.zeros(p)
            y[:first_neg] = s[:first_neg] + (
                a_val - np.sum(s[:first_neg])) / first_neg

    return(y[invperm])


def proj_sparse():

    return


if __name__ == 'main':

    s_vec = np.array([1,3,5,10])
    a_val = 20
    u_b = 6

    yy = one_proj_sorted(s_vec, a_val, u_b=u_b)
