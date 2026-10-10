import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MH_MCMC import run_MH_MCMC
from Pantheon_Data import load_data
from chain_data_loader import load_chains_data
from convergence_plots import plot_logL_multiple, plot_parameters_multiple


def chain_runs_MH_MCMC(chains, steps, epsilon, seed = 47):

    np.random.seed(seed)
    for chain in range(chains):
        run = run_MH_MCMC(z, mu_obs, inv_C, steps, epsilon)
        np.savetxt(f"MH_MCMC_chain_{chain}.txt", run)

z, mu_obs, inv_C = load_data()
chain_runs_MH_MCMC(5, 5000, 0.0142)
all_H0, all_omega_m, all_log_likelihoods, all_acceptances = load_chains_data(5, "MH_MCMC")
max_burn_in = plot_logL_multiple(all_log_likelihoods)
plot_parameters_multiple(all_H0, all_omega_m, burn_in = max_burn_in)

