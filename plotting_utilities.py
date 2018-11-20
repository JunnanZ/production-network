# Code for the paper "Equilibrium in a Model of Production Networks" by Meng
# Yu and Junnan Zhang
#
# File name: plotting_utilities.py
#
# Plotting utilities for the network graphs.
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
import networkx as nx
from graph_builder import build_dict_and_graph


def get_value_added(ps, scale_factor=2000):

    firms, G = build_dict_and_graph(ps)

    vas = []
    for firm in firms.values():
        vas.append(firm.va * scale_factor)

    return vas


def get_graph(firms, G, color='#4f79e290', scale_factor=10, sep='6:2',
              figsizes=(10, 10), va='cg'):

    vas = []
    scale = scale_factor
    if va == 'cg':
        for firm in firms.values():
            vas.append(np.sqrt(firm.va)*scale)
    elif va == 'c':
        for firm in firms.values():
            vas.append(np.sqrt(firm.va_c)*scale)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(root='1', size=figsizes, ranksep=sep, margin=0)
    for i in A.nodes():
        n = A.get_node(i)
        n.attr['width'] = vas[int(i)-1]
    A.node_attr.update(label=' ', shape='circle', style='filled',
                       fillcolor=color, penwidth=0.4)
    A.edge_attr.update(penwidth=0.5)
    return A
