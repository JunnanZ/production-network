# Code for the paper "Equilibrium in a Model of Production Networks" by Meng
# Yu and Junnan Zhang
#
# File name: graph_builder.py
#
# For a given parameterization of the model, use recursion to build a
# dictionary that gives the structure of the resulting network.
#
# The code is based on the code by John Stachurski for the paper "Span of
# Control, Transaction Costs and the Structure of Production Chains".
#
# For example usages, see 'plots.py'.
#
# Author: Junnan Zhang
#
# Date: Nov 18, 2018

import networkx as nx
from scipy.stats import poisson


class Firm:

    def __init__(self, s, param, ell):

        self.s = s
        self.param = param
        self.ell = ell
        self.subcontractors = []

    def set_va(self, va, va_c):
        self.va = va
        self.va_c = va_c

    def print(self):
        out = "value added {} and subcontractors ".format(self.va)
        print(out)
        print(self.subcontractors)


def build_dict(ps, verbose=False, tol=1e-2):
    level = 1
    num_firms_at_this_level = 1
    current_firm_num = 1
    first_firm_at_level = 1

    firms = {}
    fmin, param_star, ell_star = ps.solve_min(ps.p_func, 1)
    firms[1] = Firm(1, param_star, ell_star)

    def gen_k(param_star):
        if ps.dist == 'poisson':
            k_star = poisson.rvs(param_star, loc=1)
        elif ps.dist == 'discreet':
            k_star = param_star
        return k_star

    while True:
        current_mark = current_firm_num
        for n in range(num_firms_at_this_level):
            no = first_firm_at_level + n
            param, ell = firms[no].param, firms[no].ell
            k = gen_k(param)
            # print("k = ", k)
            va = ps.c(ell) + ps.g(k - 1)
            va_c = ps.c(ell)
            firms[no].set_va(va, va_c)

            s = firms[no].s
            if s == ell or s < tol:
                continue

            # Otherwise add subcontractors
            s_next = (s - ell)/k
            fmin, param_star, ell_star = ps.solve_min(ps.p_func, s_next)
            for k in range(k):
                current_firm_num += 1
                firms[no].subcontractors.append(current_firm_num)
                firms[current_firm_num] = Firm(s_next, param_star, ell_star)

        # == next level values == #
        first_firm_at_level = first_firm_at_level + num_firms_at_this_level
        level += 1
        num_firms_at_this_level = current_firm_num - current_mark

        if num_firms_at_this_level == 0:
            break

    return firms


def build_dict_and_graph(ps, verbose=False, tol=1e-2):
    firms = build_dict(ps, verbose=verbose, tol=tol)
    G = nx.Graph()

    for firm_no, firm in firms.items():
        for sub in firm.subcontractors:
            G.add_edge(firm_no, sub)
    return firms, G
