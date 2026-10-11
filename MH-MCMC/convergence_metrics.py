import numpy as np

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
