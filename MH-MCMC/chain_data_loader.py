import numpy as np

def load_chains_data(number_chains, chain_name):
    all_H0, all_omega_m, all_log_likelihoods, all_acceptances = [], [], [], []
    for i in range(number_chains):
        chain = np.loadtxt(f"{chain_name}_chain_{i}.txt")
        all_H0.append(chain[:, 0])
        all_omega_m.append(chain[:, 1])
        all_log_likelihoods.append(chain[:, 2])
        all_acceptances.append(chain[:, 3])
    all_H0, all_omega_m, all_log_likelihoods, all_acceptances = np.array(all_H0), np.array(all_omega_m), np.array(all_log_likelihoods), np.array(all_acceptances)
    return all_H0, all_omega_m, all_log_likelihoods, all_acceptances
