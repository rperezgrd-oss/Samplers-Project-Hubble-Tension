import numpy as np
import matplotlib.pyplot as plt
from burn_in_checks import estimate_global_burnin
from chain_data_loader import load_chains


CMAP = plt.get_cmap('inferno')
C_BURN = '0.5'
STYLE = {'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
         'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8,
         'text.usetex': True, 'font.family': 'serif', 'mathtext.fontset': 'cm',
         'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
         'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 
         'ytick.minor.visible': True, 'savefig.bbox': 'tight', 'savefig.dpi': 1000}

plt.rcParams.update(STYLE)
H0_LABEL = r'$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$'
OM_LABEL = r'$\Omega_m$'


def chain_colors(n):
    return [CMAP(v) for v in np.linspace(0.12, 0.85, n)]


def add_shade_burnin(ax, burn_in):
    ax.axvspan(0, burn_in, color = C_BURN, alpha = 0.12, lw = 0, zorder = 0)
    ax.axvline(burn_in, color = '0.2', lw = 1.0, zorder = 3)


def plot_logL_multiple(all_log_likelihoods, n_show = 1000, linthresh = 100, ylim = (-2e4, -400), fname = 'logL_chains.pdf'):
    max_burn_in, median_burn_in = estimate_global_burnin(all_log_likelihoods)
    colors = chain_colors(len(all_log_likelihoods))

    fig, ax = plt.subplots(figsize = (6.6, 3.6), layout = 'constrained')
    add_shade_burnin(ax, max_burn_in)
    ax.axvline(median_burn_in, color = '0.2', ls = '--', lw = 0.9, zorder = 3)
    for i, (c, logL) in enumerate(zip(colors, all_log_likelihoods)):
        ax.plot(logL[:n_show], color = c, lw = 0.8, alpha = 0.9, label = f'Chain {i + 1}')
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.01),
              ncol = min(len(all_log_likelihoods), 6), frameon = False,
              handlelength = 1.4, columnspacing = 1.2)

    for x, text, ha, dx in ((max_burn_in, f'max burn-in = {max_burn_in}', 'left', 4),
                            (median_burn_in, f'median = {median_burn_in}', 'right', -4)):
        ax.annotate(text, xy = (x, 0.03), xycoords = ('data', 'axes fraction'),
                    xytext = (dx, 0), textcoords = 'offset points',
                    ha = ha, va = 'bottom', fontsize = 8, color = '0.2')

    ax.set_yscale('symlog', linthresh = linthresh)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, n_show)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\ln\mathcal{L}(\theta)$')
    ax.grid(which = 'major', color = '0.9', lw = 0.6)
    ax.set_axisbelow(True)
    fig.savefig(fname)
    plt.show()
    return max_burn_in


def plot_parameters_multiple(all_H0, all_omega_m, burn_in = None, n_show = 1000, fname = 'Parameter_chains.pdf'):
    colors = chain_colors(len(all_H0))

    fig, (a_H0, a_om) = plt.subplots(2, 1, figsize = (6.6, 3.6), sharex = True, gridspec_kw = dict(hspace = 0))
    for i, (c, H0, om) in enumerate(zip(colors, all_H0, all_omega_m)):
        a_H0.plot(H0[:n_show], color = c, lw = 0.8, alpha = 0.9, label = f'Chain {i + 1}')
        a_om.plot(om[:n_show], color = c, lw = 0.8, alpha = 0.9)

    for ax in (a_H0, a_om):
        if burn_in is not None:
            add_shade_burnin(ax, burn_in)
        ax.grid(which = 'major', color = '0.9', lw = 0.6)
        ax.set_axisbelow(True)
    a_H0.tick_params(which = 'both', labelbottom = False)
    a_H0.set_ylabel(H0_LABEL)
    a_om.set_ylabel(OM_LABEL)
    a_om.set_xlabel('Iteration')
    a_om.set_xlim(0, n_show)
    fig.align_ylabels([a_H0, a_om])
    a_H0.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.01), ncol = min(len(all_H0), 6), frameon = False, handlelength = 1.4, columnspacing = 1.2)
    if burn_in is not None:
        a_om.annotate(f'burn-in = {burn_in}', xy = (burn_in, 0.05),
                      xycoords = ('data', 'axes fraction'), xytext = (4, 0),
                      textcoords = 'offset points', fontsize = 8, color = '0.2')
    fig.savefig(fname)
    plt.show()
