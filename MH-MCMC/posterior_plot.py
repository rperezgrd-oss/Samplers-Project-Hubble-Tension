import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorsys
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'text.usetex': True,
    'font.family': 'serif',
})




def GR_from_chains(all_H0, all_omega_m, burn_in):
    chains_params = [np.column_stack([np.array(all_omega_m[i])[burn_in:], np.array(all_H0[i])[burn_in:]]) for i in range(len(all_omega_m))]
    N = min(len(p) for p in chains_params)
    chains_params = [p[:N] for p in chains_params]
    n_params = chains_params[0].shape[1]
    R_hats = np.zeros(n_params)
    labels = [r'$\Omega_m$', r'H$_0$']

    for k in range(n_params):
        samples = [chain[:, k] for chain in chains_params]
        chain_means = np.array([s.mean() for s in samples])
        chain_vars = np.array([s.var(ddof=1) for s in samples])
        W = np.mean(chain_vars)
        B = N * np.var(chain_means, ddof=1)
        V_hat = (N - 1) / N * W + B / N
        R_hats[k] = np.sqrt(V_hat / W)
        print(f"R_hat {labels[k]} = {R_hats[k]:.4f} "f"{'✅ converged' if R_hats[k] < 1.1 else '❌ not converged'}")

    return R_hats


#GR_values = GR_from_chains(H0_df, omega_m_df, burn_in)



def compute_acf(samples, max_lag = 200):
    samples = np.array(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    N = len(samples)
    acf = np.zeros(max_lag)
    for k in range(max_lag):
        acf[k] = np.mean((samples[:N-k] - mean) * (samples[k:] - mean)) / var
    return acf


def plot_acf(all_H0, all_omega_m, burn_in):
    colors = distinct_colors(len(all_omega_m))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4))
    axes = axes.flatten()
    H0_acfs, omega_m_acfs = [], []
    for i, (H0_chain, omega_m_chain) in enumerate(zip(all_H0, all_omega_m)):
        sample_H0, sample_omega_m = H0_chain[burn_in:], omega_m_chain[burn_in:]
        acf_H0, acf_omega_m = compute_acf(sample_H0), compute_acf(sample_omega_m)
        H0_acfs.append(acf_H0)
        omega_m_acfs.append(acf_omega_m)
        axes[0].plot(acf_H0, color=colors[i], linewidth=1.0, label=f'Chain {i+1}')
        axes[1].plot(acf_omega_m, color=colors[i], linewidth=1.0, label=f'Chain {i+1}')
    axes[0].set_xlabel('Lag $k$')
    axes[0].set_ylabel(r'Autocorrelation')
    axes[0].set_title(r'ACF - $\Omega_m$')
    axes[0].set_ylim(-0.2, 1.0)
    axes[0].grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig('acf_MH.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    
    mean_acf_omega = np.mean(omega_m_acfs, axis=0)
    mean_acf_H0 = np.mean(H0_acfs, axis=0)
    std_acf_omega = np.std(omega_m_acfs, axis=0)
    std_acf_H0 = np.std(H0_acfs, axis=0)
    plt.figure(figsize=(6, 4))
    lags = np.arange(len(mean_acf_omega))
    plt.plot(lags, mean_acf_omega, color="darkorange", linewidth=1.0, label=f'$\Omega_m$')
    plt.plot(lags, mean_acf_H0, color="blue", linewidth=1.0, label=f'$H0$')
    plt.fill_between(lags,
                 mean_acf_omega - std_acf_omega,
                 mean_acf_omega + std_acf_omega,
                 alpha=0.3, color = "darkorange")
    plt.fill_between(lags,
                 mean_acf_H0 - std_acf_H0,
                 mean_acf_H0 + std_acf_H0,
                alpha = 0.3, color = "blue")
    plt.xlabel("Lag $k$")
    plt.ylabel(r"Autocorrelation")
    plt.ylim(top = 1.0)
    plt.grid(alpha=0.3)
    plt.legend(title='Parameters',
               title_fontproperties={'weight': 'bold', 'size': 9})
    plt.show()


#plot_acf(H0_df, omega_m_df, burn_in)

clean_H0 = []
clean_omega_m = []
for i in range(5):
    clean_H0.append(H0_df[i][burn_in:])
    clean_omega_m.append(omega_m_df[i][burn_in:])

clean_H0 = np.array(clean_H0)
clean_omega_m = np.array(clean_omega_m)
clean_H0 = clean_H0.flatten()
clean_omega_m = clean_omega_m.flatten()

# Pool all chains after burn-in: one row per sample, columns (Omega_m, H0)
samples = np.vstack([np.column_stack([clean_H0, clean_omega_m])])

Sigma_hat = np.cov(samples, rowvar=False)     # 2x2 covariance matrix
R = np.corrcoef(samples, rowvar=False)        # 2x2 correlation matrix

sig = np.sqrt(np.diag(Sigma_hat))             # sigma(Omega_m), sigma(H0)
rho = R[0, 1]                                 # correlation coefficient
print(f'sigma_Om = {sig[0]:.4f}, sigma_H0 = {sig[1]:.3f}, rho = {rho:+.3f}')

"""Posterior figures for the (Omega_m, H0) MCMC run.

Both figures share one style, one Planck reference and one set of helpers, so
they look like they belong to the same paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------
# Shared constants and style
# --------------------------------------------------------------------------
PLANCK = dict(om=0.315, om_err=0.007, H0=67.4, H0_err=0.5)   # Planck 2018
CMAP = plt.get_cmap('inferno')
C_PLANCK = '#4cc9f0'          # cyan: stands out against inferno and on white
C_POST = CMAP(0.58)           # posterior colour for 1D panels, taken from the 2D map
C_POST_DARK = CMAP(0.30)
H0_LABEL = r'$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$'
OM_LABEL = r'$\Omega_m$'

STYLE = {
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 9.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8,
    'text.usetex': True, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 
    'ytick.minor.visible': True, 'savefig.bbox': 'tight', 'savefig.dpi': 1000,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _density(kde, xlim, ylim, n=250):
    """Evaluate a 2D KDE on an n x n grid."""
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def _credible_levels(Z, probs=(0.95, 0.68)):
    """Density thresholds enclosing the given posterior mass (ascending)."""
    z = np.sort(Z.ravel())[::-1]
    cdf = np.cumsum(z) / z.sum()
    return [z[np.searchsorted(cdf, p)] for p in probs]


def _window(s, pad=0.25):
    """Plot range from robust percentiles, so a few stray samples don't set it."""
    lo, hi = np.percentile(s, [0.1, 99.9])
    d = pad * (hi - lo)
    return lo - d, hi + d


def _summary(s):
    """Median and asymmetric 68% interval."""
    lo, med, hi = np.percentile(s, [16, 50, 84])
    return med, med - lo, hi - med


def _fmt(med, lo, hi, nd):
    return rf'{med:.{nd}f}^{{+{hi:.{nd}f}}}_{{-{lo:.{nd}f}}}'


def _dark_axes(ax):
    ax.set_facecolor(CMAP(0.0))
    ax.tick_params(which='both', color='white')



def distributions_plot(H0_samples, omega_samples, fname='corner_MH.pdf', label='MH-MCMC'):
    H0 = np.asarray(H0_samples, dtype=float)
    om = np.asarray(omega_samples, dtype=float)
    xlim, ylim = _window(om, 0.12), _window(H0, 0.12)

    # 2D posterior
    kde = gaussian_kde(np.vstack([om, H0]))
    X, Y, Z = _density(kde, xlim, ylim, n=220)
    Z /= Z.max()
    cl = _credible_levels(Z)

    # 1D marginals
    stats = {'om': _summary(om), 'H0': _summary(H0)}

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(6.6, 6.6))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[3.3, 1], height_ratios=[1, 3.3],
                      hspace=0.04, wspace=0.04)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_2d = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_2d)
        ax_key = fig.add_subplot(gs[0, 1])
        ax_key.axis('off')

        # ---- joint posterior ------------------------------------------------
        _dark_axes(ax_2d)
        ax_2d.contourf(X, Y, np.ma.masked_less(Z, 5e-3),
                       levels=np.linspace(5e-3, 1, 24), cmap=CMAP)
        cs = ax_2d.contour(X, Y, Z, levels=cl, colors='white',
                           linewidths=[0.8, 1.2], linestyles=['--', '-'])
        ax_2d.clabel(cs, fmt={cl[0]: '95\%', cl[1]: '68\%'}, fontsize=8,
                     colors='white', inline_spacing=3)
        ax_2d.plot(stats['om'][0], stats['H0'][0], marker='+', color='white',
                   ms=9, mew=1.2, zorder=5)

        # Planck: a cross-hair where it is in range, an arrow where it is not
        if xlim[0] < PLANCK['om'] < xlim[1]:
            ax_2d.axvline(PLANCK['om'], color=C_PLANCK, ls=':', lw=1.1)
        if ylim[0] < PLANCK['H0'] < ylim[1]:
            ax_2d.axhline(PLANCK['H0'], color=C_PLANCK, ls=':', lw=1.1)
        else:
            x_arrow = np.clip(PLANCK['om'], *xlim)
            ax_2d.annotate(rf'Planck $H_0={PLANCK["H0"]}$', color=C_PLANCK, fontsize=7.5,
                           xy=(x_arrow, ylim[0]), xytext=(6, 22), textcoords='offset points',
                           ha='left', va='bottom',
                           arrowprops=dict(arrowstyle='-|>', color=C_PLANCK, lw=1.0,
                                           shrinkA=0, shrinkB=0, mutation_scale=8))
        ax_2d.set(xlim=xlim, ylim=ylim, xlabel=OM_LABEL, ylabel=H0_LABEL)

        # ---- marginals --------------------------------------------------------
        def marginal(ax, s, lim, key, vertical):
            med, lo, hi = stats[key]
            g = np.linspace(*lim, 400)
            p = gaussian_kde(s)(g)
            inside = (g >= med - lo) & (g <= med + hi)
            if vertical:   # density along x, parameter along y
                ax.hist(s, bins=40, range=lim, density=True, orientation='horizontal',
                        histtype='stepfilled', color=C_POST, alpha=0.3, lw=0)
                ax.fill_betweenx(g, 0, p, where=inside, color=C_POST, alpha=0.45, lw=0)
                ax.plot(p, g, color=C_POST_DARK, lw=1.4)
                ax.axhline(med, color=C_POST_DARK, ls='--', lw=1.0)
                ax.set_xlim(0, p.max() * 1.15)
            else:
                ax.hist(s, bins=40, range=lim, density=True,
                        histtype='stepfilled', color=C_POST, alpha=0.3, lw=0)
                ax.fill_between(g, 0, p, where=inside, color=C_POST, alpha=0.45, lw=0)
                ax.plot(g, p, color=C_POST_DARK, lw=1.4)
                ax.axvline(med, color=C_POST_DARK, ls='--', lw=1.0)
                ax.set_ylim(0, p.max() * 1.15)

        marginal(ax_top, om, xlim, 'om', vertical=False)
        marginal(ax_right, H0, ylim, 'H0', vertical=True)

        if xlim[0] < PLANCK['om'] < xlim[1]:
            ax_top.axvspan(PLANCK['om'] - PLANCK['om_err'], PLANCK['om'] + PLANCK['om_err'],
                           color=C_PLANCK, alpha=0.25, lw=0)
            ax_top.axvline(PLANCK['om'], color=C_PLANCK, ls=':', lw=1.1)

        # Summary values as panel titles (corner.py convention)
        ax_top.set_title(rf'$\Omega_m = {_fmt(*stats["om"], 3)}$', pad=6, fontsize=10)
        ax_right.text(0.97, 0.985, rf'$H_0 = {_fmt(*stats["H0"], 2)}$',
                      transform=ax_right.transAxes, ha='right', va='top', fontsize=10)

        # Clean marginal axes: no density ticks, no tick labels shared with the 2D panel
        ax_top.tick_params(labelbottom=False, left=False, labelleft=False,
                           right=False, top=False, which='both')
        ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False,
                             top=False, right=False, which='both')
        for side in ('top', 'right', 'left'):
            ax_top.spines[side].set_visible(False)
        for side in ('top', 'right', 'bottom'):
            ax_right.spines[side].set_visible(False)

        # ---- key in the empty corner ----------------------------------------
        handles = [
            Line2D([], [], color='0.15', lw=1.2, label='68\% credible region'),
            Line2D([], [], color='0.15', lw=0.8, ls='--', label='95\% credible region'),
            Line2D([], [], color='0.15', marker='+', ls='none', ms=8, mew=1.2,
                   label='Posterior median'),
            Patch(color=C_POST, alpha=0.55, label='68\% interval (1D)'),
            Line2D([], [], color=C_PLANCK, ls=':', lw=1.3, label='Planck 2018'),
        ]
        ax_key.legend(handles=handles, loc='upper right', frameon=True,
                      handlelength=1.8, borderaxespad=0, labelspacing=0.55)

        fig.savefig(fname)
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Full prior range + zoom (supplementary figure)
# --------------------------------------------------------------------------
def full_range_plot(H0_samples, omega_samples, full_lim=((0.1, 0.9), (50, 100)),
                    bw_method=None, floor=1e-3,
                    fname='posterior_full_range_supplementary.pdf'):
    H0 = np.asarray(H0_samples, dtype=float)
    om = np.asarray(omega_samples, dtype=float)
    kde = gaussian_kde(np.vstack([om, H0]), bw_method=bw_method)
    zx, zy = _window(om, 0.3), _window(H0, 0.3)   # box in (a) == extent of (b)

    Xf, Yf, Zf = _density(kde, *full_lim)
    Xz, Yz, Zz = _density(kde, zx, zy)
    peak = max(Zf.max(), Zz.max())                # one normalisation -> one colorbar
    Zf, Zz = Zf / peak, Zz / peak
    cl = _credible_levels(Zz)

    filled = dict(levels=np.logspace(np.log10(floor), 0, 13),
                  norm=LogNorm(floor, 1), cmap=CMAP)
    ci = dict(levels=cl, colors='white', linewidths=[0.8, 1.1], linestyles=['--', '-'])

    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 7.2), layout='constrained')
        for ax in (a1, a2):
            _dark_axes(ax)
            ax.set(xlabel=OM_LABEL, ylabel=H0_LABEL)

        # (a) full prior range
        cf = a1.contourf(Xf, Yf, np.ma.masked_less(Zf, floor), **filled)
        a1.contour(Xf, Yf, Zf, **ci)
        a1.axvline(PLANCK['om'], color=C_PLANCK, ls=':', lw=0.9, alpha=0.8)
        a1.axhline(PLANCK['H0'], color=C_PLANCK, ls=':', lw=0.9, alpha=0.8)
        a1.errorbar(PLANCK['om'], PLANCK['H0'], xerr=PLANCK['om_err'], yerr=PLANCK['H0_err'],
                    fmt='*', ms=9, color=C_PLANCK, mec='k', mew=0.4, zorder=5)
        a1.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                               fill=False, ec='white', lw=0.9))
        a1.set(xlim=full_lim[0], ylim=full_lim[1])

        # (b) zoom
        a2.contourf(Xz, Yz, np.ma.masked_less(Zz, floor), **filled)
        cs = a2.contour(Xz, Yz, Zz, **ci)
        a2.clabel(cs, fmt={cl[0]: '95\%', cl[1]: '68\%'}, fontsize=8, colors='white')
        a2.plot(np.median(om), np.median(H0), '+', color='white', ms=9, mew=1.2)
        if zx[0] < PLANCK['om'] < zx[1]:
            a2.axvline(PLANCK['om'], color=C_PLANCK, ls=':', lw=1.1)
        if not zy[0] < PLANCK['H0'] < zy[1]:
            a2.annotate(rf'Planck $H_0={PLANCK["H0"]}$', color=C_PLANCK, fontsize=8,
                        xy=(np.clip(PLANCK['om'], *zx), zy[0]), xytext=(6, 22),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='-|>', color=C_PLANCK, lw=1.0,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
        a2.set(xlim=zx, ylim=zy)

        for x in zx:   # zoom connectors
            con = ConnectionPatch(xyA=(x, zy[0]), coordsA=a1.transData,
                                  xyB=(x, zy[1]), coordsB=a2.transData,
                                  color='0.55', lw=0.6, ls='--')
            con.set_in_layout(False)
            fig.add_artist(con)

        for ax, tag in ((a1, '(a)'), (a2, '(b)')):
            ax.text(0.02, 0.96, tag, transform=ax.transAxes, va='top',
                    color='white', fontweight='bold')

        handles = [
            Line2D([], [], marker='*', ls=':', color=C_PLANCK, ms=9, mec='k', mew=0.4,
                   label=r'Planck 2018: $\Omega_m=0.315,\ H_0=67.4$'),
            Line2D([], [], color='white', lw=1.1, label='68\% credible region'),
            Line2D([], [], color='white', lw=0.8, ls='--', label='95\% credible region'),
            Line2D([], [], color='white', lw=0.9, label='Zoom window (b)'),
        ]
        a1.legend(handles=handles, loc='upper right', framealpha=0.85,
                  facecolor='0.12', edgecolor='0.4', labelcolor='white')

        cb = fig.colorbar(cf, ax=[a1, a2], shrink=0.5, aspect=30, pad=0.02)
        cb.set_label('Relative posterior density')
        cb.set_ticks([1e-3, 1e-2, 1e-1, 1])

        fig.savefig(fname)
        plt.show()
    return fig


if __name__ == '__main__':
    distributions_plot(clean_H0, clean_omega_m, fname='demo_corner.pdf')
    full_range_plot(clean_H0, clean_omega_m, fname='demo_full.pdf')
