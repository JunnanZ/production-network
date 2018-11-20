# Code for the paper "Equilibrium in a Model of Production Networks" by Meng
# Yu and Junnan Zhang
#
# File name: plots.py
#
# Generate figures in the paper.
#
# Author: Junnan Zhang
#
# Date: Nov 18, 2018

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
from matplotlib import rcParams
from stochastic_k import RP
from plotting_utilities import get_graph
from graph_builder import build_dict_and_graph

rc('text', usetex=True)
rc('font', family='serif')

color_r = '#ed444d'
color_b = '#4f79e2'
color_g = '#49c070'
color_text = '#080808'
r_list = [color_r + a for a in ['50', '40', '30', '20', '10']]
b_list = [color_b + a for a in ['50', '40', '30', '20', '10']]
g_list = [color_g + a for a in ['50', '40', '30', '20', '10']]


# Figure 1: price function example
n = 50
ps = RP(n=n, g=lambda k: 50*k, delta=10, c=lambda x: np.exp(10*x) - 1)
ps.set_prices2()
plt.figure(figsize=(8, 3.5))
plt.plot(ps.grid, ps.p, linestyle='-', color=color_b, alpha=0.5)
plt.xlabel('production stage (s)')
plt.ylabel(r'price function $p^*(s)$')
plt.savefig('price_example.pdf', bbox_inches='tight', pad_inches=0)
plt.close()


# Figure 2: price function comparison
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
fig.subplots_adjust(wspace=0.1)
n = 5000
ps = RP(n=n, g=lambda k: 50*k, delta=10, c=lambda x: np.exp(10*x) - 1,
        kmax=10)
ps.set_prices2()
ax[0].plot(ps.grid, ps.p, linestyle='-', color=color_b, alpha=0.5,
           label=r'$\delta=10$')
ax[1].plot(ps.grid, ps.p, linestyle='-', color=color_b, alpha=0.5,
           label=r'$\beta=50$')

ps = RP(n=n, g=lambda k: 50*k, delta=15, c=lambda x: np.exp(10*x) - 1,
        kmax=10)
ps.set_prices2()
ax[0].plot(ps.grid, ps.p, linestyle='--', color='#ed444d', alpha=0.5,
           label=r'$\delta=15$')

ps = RP(n=n, g=lambda k: 100*k, delta=10, c=lambda x: np.exp(10*x) - 1,
        kmax=10)
ps.set_prices2()
ax[1].plot(ps.grid, ps.p, linestyle='--', color='#ed444d', alpha=0.5,
           label=r'$\beta=100$')

ax[0].set_xlabel('production stage (s)')
ax[1].set_xlabel('production stage (s)')
ax[0].set_ylabel(r'price function $p^*(s)$')
ax[1].tick_params(axis='y', left=False)
ax[0].legend()
ax[1].legend()

plt.savefig('price_comp.pdf', bbox_inches='tight', pad_inches=0)


# Figure 3: computation time comparison
def err(n, p_acc_func, p):
    p_acc = p_acc_func(np.linspace(0, 1, n))
    return np.max(np.abs(p_acc[1:] - p[1:]))


def compute_acc(g, delta, c):
    ps = RP(n=50000, g=g, delta=delta, c=c)
    ps.set_prices2()
    return ps.p_func


beta_list = [1, 0.01, 0.01, 0.01, 0.05]
func_list = [lambda x: np.exp(10*x) - 1, lambda x: np.exp(x) - 1,
             lambda x: np.exp(x**2) - 1, lambda x: x**2 + x,
             lambda x: x**2 + np.exp(x) - 1]
time_1 = []
time_2 = []
for delta in [1.1, 1.01]:
    for i in range(len(beta_list)):
        beta = beta_list[i]
        c = func_list[i]
        p_acc = compute_acc(lambda k: beta*k, delta, c)

        n = 1000
        ps = RP(n=n, g=lambda k: beta*k, delta=delta, c=c)

        t0 = timer()
        ps.set_prices()
        t1 = timer()
        time_1.append(t1 - t0)
        print(err(n, p_acc, ps.p))

        t0 = timer()
        ps.set_prices2()
        t1 = timer()
        time_2.append(t1 - t0)
        print(err(n, p_acc, ps.p))

# results
time_1 = [27.875319243990816, 8.296202068013372, 47.287453513010405,
          7.165919941995526, 12.85498929201276, 131.11969668898382,
          18.82582350500161, 59.87139473500429, 19.379346911999164,
          43.91938027497963]
time_2 = [3.2009800499945413, 2.7448071209946647, 3.0622648729768116,
          2.4626454260142054, 2.645818232995225, 3.3310844369989354,
          2.755331556982128, 3.8060924599994905, 2.150440797995543,
          2.478869067010237]

# plot bar graphs
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(len(time_1))
bar_width = 0.35
op = 0.5
rcParams['hatch.linewidth'] = 1.2
g1 = ax.bar(index, time_1, bar_width, alpha=op, color=color_b, hatch='//',
            edgecolor='w', linewidth=0, label='Method 1')
g2 = ax.bar(index+bar_width, time_2, bar_width, alpha=op, color='#ed444d',
            label='Method 2')
rc('text', usetex=True)
rc('font', family='serif')
ax.set_xticks(index + bar_width / 2)
xticks_label = np.tile(np.arange(1, len(time_1)/2+1), 2)
xticks_label = xticks_label.astype(int)
ax.set_xticklabels(xticks_label)
ax.set_xlabel('Model')
ax.set_ylabel('Computation time (s)')
mid = len(time_1)/2 - 0.5 + bar_width/2
ax.axvline(x=mid, linestyle='--', linewidth=0.6, color='k')
ax.text(mid/2, 100, r'$\delta = 1.1$', ha='center')
ax.text(3*mid/2, 100, r'$\delta = 1.01$', ha='center')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.legend()
fig.tight_layout()
plt.savefig('speed.pdf', bbox_inches='tight', pad_inches=0)
plt.close()


# Figure 4: network graphs
ps = RP(n=100, g=lambda k: 0.0005*k**1.5, delta=1.05,
        c=lambda x: x**1.2, dist='poisson')
ps.set_prices2()
np.random.seed(1233)
firms1, G1 = build_dict_and_graph(ps)

A = get_graph(firms1, G1, sep='6:2')
A.draw('random_network_1.pdf', prog='twopi')

ps = RP(n=100, g=lambda k: 0.0005*k**1.5, delta=1.1,
        c=lambda x: x**1.2, dist='poisson')
ps.set_prices2()
np.random.seed(1236)
firms2, G2 = build_dict_and_graph(ps)

A = get_graph(firms2, G2, sep='6:2')
A.draw('random_network_2.pdf', prog='twopi')

ps = RP(n=100, g=lambda k: 0.0001*k**1.5, delta=1.05,
        c=lambda x: x**1.2, dist='poisson')
ps.set_prices2()
np.random.seed(1233)
firms3, G3 = build_dict_and_graph(ps)

A = get_graph(firms3, G3, sep='6:2')
A.draw('random_network_3.pdf', prog='twopi')

ps = RP(n=100, g=lambda k: 0.0005*k**1.5, delta=1.05,
        c=lambda x: x**1.15, dist='poisson')
ps.set_prices2()
np.random.seed(1238)
firms4, G4 = build_dict_and_graph(ps)

A = get_graph(firms4, G4, sep='6:2')
A.draw('random_network_4.pdf', prog='twopi')
