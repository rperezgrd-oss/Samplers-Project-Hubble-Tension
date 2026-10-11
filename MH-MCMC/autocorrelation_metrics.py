import numpy as np
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(1, 2, figsize = (6.6, 4))
    axes = axes.flatten()
    H0_acfs, omega_m_acfs = [], []
    for i, (H0_chain, omega_m_chain) in enumerate(zip(all_H0, all_omega_m)):
        sample_H0, sample_omega_m = H0_chain[burn_in:], omega_m_chain[burn_in:]
        acf_H0, acf_omega_m = compute_acf(sample_H0), compute_acf(sample_omega_m)
        H0_acfs.append(acf_H0)
        omega_m_acfs.append(acf_omega_m)
        axes[0].plot(acf_H0, color = colors[i], linewidth = 1.0, label = f'Chain {i+1}')
        axes[1].plot(acf_omega_m, color = colors[i], linewidth = 1.0, label = f'Chain {i+1}')
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
