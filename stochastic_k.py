# Code for the paper "Equilibrium in a Model of Production Networks" by Meng
# Yu and Junnan Zhang
#
# File name: stochastic_k.py
#
# Compute the equilibrium prices as a fixed point of T specified in equation
# (3) and (6) in the paper. Two methods are used: one is successive
# evaluations of T and the other is Algorithm 1 in the paper.
#
# The code is based on the code by John Stachurski for the paper "Span of
# Control, Transaction Costs and the Structure of Production Chains".
#
# For example usages, see 'plots.py'.
#
# Author: Junnan Zhang
#
# Date: Nov 18, 2018

import numpy as np
from scipy.optimize import fminbound
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import geom


class RP:

    def __init__(self, n=500, delta=1.01, g=lambda k: 0.1 * k,
                 c=lambda x: np.exp(3 * x) - 1, dist='discreet', kmax=30,
                 method='L-BFGS-B'):

        self.n = n
        self.delta = delta
        self.g = g
        self.c = c
        self.grid = np.linspace(0, 1, num=n)
        self.dist = dist
        if dist == 'discreet':
            self.solve_min = self.solve_min_dis
        else:
            self.solve_min = self.solve_min_st
        self.kmax = kmax
        self.method = method

    def set_prices(self):
        self.p = self.compute_prices()
        self.p_func = lambda x: np.interp(x, self.grid, self.p)

    def set_prices2(self):
        self.p = self.compute_prices2()
        self.p_func = lambda x: np.interp(x, self.grid, self.p)

    def solve_min_dis(self, p, s, s2=None):
        """
        Solve for minimum and minimizers when at stage s, given p.

        The parameter p should be supplied as a function.

        """
        current_function_min = np.inf
        delta, g, c, kmax = self.delta, self.g, self.c, self.kmax
        if s2 is None:
            s2 = s

        for k in range(1, kmax+1):
            def Tp(ell):
                return (delta * k * p((s - ell)/k) + c(ell) + g(k - 1))

            ell_star_eval_at_k = fminbound(Tp, s-s2, s)
            function_value = Tp(ell_star_eval_at_k)

            if function_value < current_function_min:
                current_function_min = function_value
                k_star = k
                ell_star = ell_star_eval_at_k
            # else:
            #     break
        # if k_star == kmax:
        #     print("Warning: kmax reached.")
        return current_function_min, k_star, ell_star

    def solve_min_st(self, p, s, s2=None):
        """
        Same as the previous function but in the stochastic case.
        """
        delta, g, c, kmax = self.delta, self.g, self.c, self.kmax
        dist, E, method = self.dist, self.E, self.method
        if s2 is None:
            s2 = s

        def fun(x):
            # x[0]: s-t; x[1]: param
            return E(lambda k: delta * k * p((s - x[0])/k) + c(x[0]) +
                     g(k - 1), x[1])

        if dist == 'poisson':
            bnds = ((s-s2, s), (0, kmax))
            x0 = 0
        elif dist == 'geom':
            bnds = ((s-s2, s), (1e-6, 1))
            x0 = 1

        res = minimize(fun, (s, x0), method=method, bounds=bnds)
        return res.fun, res.x[1], res.x[0]

    def apply_T(self, current_p):
        n = self.n
        solve_min = self.solve_min

        def p(x): return np.interp(x, self.grid, current_p)
        new_p = np.empty(n)

        for i, s in enumerate(self.grid):
            current_function_min, param_star, ell_star = solve_min(p, s)
            new_p[i] = current_function_min

        return new_p

    def E(self, f, param, n=100):
        dist = self.dist
        if dist == 'poisson':
            def h(k): return f(k) * poisson.pmf(k, param, loc=1)
        elif dist == 'geom':
            def h(k): return f(k) * geom.pmf(k, param)
        else:
            print("Distribution error.")
        return sum(map(h, np.arange(1, n+1)))

    def compute_prices(self, tol=1e-3, verbose=False):
        """
        Iterate with T. The initial condition is p = c.
        """
        c = self.c
        current_p = c(self.grid)  # Initial condition is c
        error = tol + 1
        n = 0
        while error > tol:
            new_p = self.apply_T(current_p)
            error = np.max(np.abs(current_p - new_p))
            if verbose is True:
                print(error)
            current_p = new_p
            n += 1
        print(n)
        return new_p

    def compute_prices2(self):
        """
        Compute the price vector using the algorithm specified in the
        paper.

        """
        grid, n, c, kmax = self.grid, self.n, self.c, self.kmax
        solve_min = self.solve_min
        new_p = np.zeros(n)
        new_p[1] = c(grid[1])

        for i in range(2, n):
            def interp_p(x): return np.interp(x, grid[:i], new_p[:i])

            p_min, param_star, ell_star = solve_min(interp_p, grid[i],
                                                    grid[i-1])
            # print(p_min, param_star, ell_star)
            if param_star >= kmax:
                raise ValueError("kmax reached")
            new_p[i] = p_min
        return new_p

    def plot_prices(self, plottype='-', label=None):
        plt.plot(self.grid, self.p, plottype, label=label)
