import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MH_MCMC import run_MH_MCMC
from Pantheon_Data import load_data


STYLE = {'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
         'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8,
         'text.usetex': True, 'font.family': 'serif', 'mathtext.fontset': 'cm',
         'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
         'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 
         'ytick.minor.visible': True, 'savefig.bbox': 'tight', 'savefig.dpi': 1000}

plt.rcParams.update(STYLE)


CMAP = plt.get_cmap('inferno')
C_DATA = CMAP(0.58)
C_DATA_DARK = CMAP(0.30)
C_PICK = '#4cc9f0'

OPTIMAL_RATES = ((0.44,  r'$0.44$  ($d=1$)'),
                 (0.35,  r'$0.35$  ($d=2$)'),
                 (0.234, r'$0.234$  ($d\to\infty$)'))


def epsilon_tuning(repetitions = 20, iterations = 100, epsilons = np.logspace(-3, -1, 10)):
    accept_rate_data = []
    for e in epsilons:
        rates = [np.mean(run_MH_MCMC(z, mu_obs, inv_C, iterations, epsilon = e)[1:, 3]) for _ in range(repetitions)]
        accept_rate_data.append([np.median(rates), *np.percentile(rates, [16, 84])])
    return epsilons, np.array(accept_rate_data)


def find_crossing(x, y, target):
    idx = np.where((y[:-1] >= target) & (y[1:] < target))[0]
    if idx.size == 0:
        return None
    i = idx[-1]
    return 10 ** np.interp(target, y[i:i + 2][::-1], np.log10(x[i:i + 2])[::-1])


def plot_epsilon_tuning(epsilons, accept_rate_data, target = 0.35, refs = OPTIMAL_RATES, n_rep = None, fname = 'Epsilon_Tuning.pdf'):
    eps = np.asarray(epsilons, dtype=float)
    med, lo, hi = np.asarray(accept_rate_data, dtype = float).T
    eps_star = find_crossing(eps, med, target)

    fig, ax = plt.subplots(figsize = (6.6, 3.6), layout = 'constrained')
    r_vals = [r for r, _ in refs]
    ax.axhspan(min(r_vals), max(r_vals), color = '0.5', alpha = 0.08, lw = 0)
    for r, text in refs:
        ax.axhline(r, color = '0.35', ls='--', lw = 0.8, zorder = 1)
        ax.text(0.995, r, text, transform = ax.get_yaxis_transform(), ha = 'right', va = 'bottom', fontsize = 8, color = '0.25')

    run = f'{n_rep} runs' if n_rep else 'repeated runs'
    ax.fill_between(eps, lo, hi, color = C_DATA, alpha = 0.30, lw = 0, label = f'16th–84th percentile over {run}')
    ax.plot(eps, med, color=C_DATA_DARK, lw = 1.5, marker = 'o', ms = 4.5, mfc = 'white', mec = C_DATA_DARK, mew = 1.2, 
            zorder = 3, label = 'Median')

    if eps_star is not None:
        ax.axvline(eps_star, color = C_PICK, ls = ':', lw = 1.2, zorder = 2)
        ax.plot(eps_star, target, marker = 'o', ms = 6, color = C_PICK, mec = 'k', mew = 0.5, zorder = 4)
        ax.annotate(rf'$\varepsilon^\ast \approx {eps_star:.4f}$', xy = (eps_star, target),
                    xytext = (10, 28), textcoords = 'offset points', fontsize = 8,
                    arrowprops=dict(arrowstyle = '-', color = '0.3', lw = 0.7, shrinkA = 0, shrinkB = 4))
        
    ax.set_xscale('log')
    ax.set_xlim(eps.min() / 1.15, eps.max() * 1.15)
    ax.set_ylim(0, np.ceil(hi.max() * 10 + 0.5) / 10)
    ax.set_xlabel(r'Proposal step size $\varepsilon$')
    ax.set_ylabel('Acceptance rate')
    ax.grid(which = 'major', color = '0.9', lw = 0.6)
    ax.set_axisbelow(True)
    ax.legend(loc = 'lower left', frameon = False)
    fig.savefig(fname)
    plt.show()


np.random.seed(37)
z, mu_obs, inv_C = load_data()
epsilons, accept_rate_data = epsilon_tuning()
np.savetxt("eps.txt", epsilons)
np.savetxt("acc.txt", accept_rate_data)

plot_epsilon_tuning(epsilons, accept_rate_data, n_rep = 20)
