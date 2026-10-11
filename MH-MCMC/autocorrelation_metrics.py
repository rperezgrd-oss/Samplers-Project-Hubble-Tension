import numpy as np
import matplotlib.pyplot as plt

# Same look as the other figures
CMAP = plt.get_cmap('inferno')
C_H0 = CMAP(0.25)      # dark purple
C_OM = CMAP(0.68)      # orange
STYLE = {
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'savefig.bbox': 'tight', 'savefig.dpi': 300,
}


def chain_colors(n):
    return [CMAP(v) for v in np.linspace(0.12, 0.85, n)]


def compute_acf(samples, max_lag=200):
    """Autocorrelation up to max_lag, via FFT (same estimator as before, much faster)."""
    x = np.asarray(samples, float) - np.mean(samples)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf[:max_lag] / acf[0]


def integrated_time(samples):
    """Integrated autocorrelation time, Geyer's initial positive sequence."""
    rho = compute_acf(samples, max_lag=len(samples))
    tau = -1.0
    for m in range(0, len(rho) - 1, 2):
        pair = rho[m] + rho[m + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return tau


def _acf_axes(ax):
    ax.axhline(0, color='0.3', lw=0.7, zorder=1)
    ax.grid(which='major', color='0.9', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'Lag $k$')


def plot_acf(all_H0, all_omega_m, burn_in, max_lag=200,
             fname='acf_MH.pdf', fname_mean='acf_mean_MH.pdf'):
    colors = chain_colors(len(all_H0))
    acfs = {'H0': [], 'om': []}
    taus = {'H0': [], 'om': []}
    for H0, om in zip(all_H0, all_omega_m):
        for key, chain in (('H0', H0[burn_in:]), ('om', om[burn_in:])):
            acfs[key].append(compute_acf(chain, max_lag))
            taus[key].append(integrated_time(chain))
    acfs = {k: np.array(v) for k, v in acfs.items()}
    lags = np.arange(max_lag)

    # ---- Figure 1: every chain, one panel per parameter ------------------------
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True, layout='constrained')
        panels = (('H0', r'$H_0$'), ('om', r'$\Omega_m$'))
        for ax, (key, name) in zip(axes, panels):
            _acf_axes(ax)
            for i, (c, acf) in enumerate(zip(colors, acfs[key])):
                ax.plot(lags, acf, color=c, lw=1.0, alpha=0.9, label=f'Chain {i + 1}')
            tau = np.mean(taus[key])
            ax.set_title(rf'{name}:  $\bar\tau = {tau:.1f}$')
            ax.set_xlim(0, max_lag - 1)
        axes[0].set_ylabel('Autocorrelation')
        axes[0].set_ylim(-0.2, 1.02)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center',
                   ncol=min(len(all_H0), 6), frameon=False,
                   handlelength=1.4, columnspacing=1.2)
        fig.savefig(fname)
        plt.show()

    # ---- Figure 2: mean ± spread over chains, both parameters ------------------
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 3.4), layout='constrained')
        _acf_axes(ax)
        for key, name, c in (('H0', r'$H_0$', C_H0), ('om', r'$\Omega_m$', C_OM)):
            mean, std = acfs[key].mean(axis=0), acfs[key].std(axis=0)
            tau = np.mean(taus[key])
            ax.fill_between(lags, mean - std, mean + std, color=c, alpha=0.22, lw=0)
            ax.plot(lags, mean, color=c, lw=1.5,
                    label=rf'{name}  ($\bar\tau = {tau:.1f}$)')
            # Mark the integrated autocorrelation time on the curve
            if tau < max_lag:
                ax.plot(tau, np.interp(tau, lags, mean), 'o', ms=4.5, color=c,
                        mec='white', mew=0.8, zorder=4)
        ax.set_xlim(0, max_lag - 1)
        ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.02)
        ax.set_ylabel('Autocorrelation')
        ax.legend(loc='upper right', frameon=False,
                  title=f'Mean ± 1σ over {len(all_H0)} chains',
                  title_fontproperties={'size': 7.5})
        fig.savefig(fname_mean)
        plt.show()

    return {k: np.mean(v) for k, v in taus.items()}


taus = plot_acf(all_H0, all_omega_m, burn_in=max_burn_in)
