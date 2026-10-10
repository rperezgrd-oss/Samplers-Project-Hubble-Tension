import numpy as np
from cosmological_formulas import theoretical_distance_modulus


def run_MH_MCMC(z, mu_obs, inv_C, iterations, epsilon = 0.01):

    """
    z: Data Redshift
    mu_obs: Observed Distance Moduli
    inv_C: Inverse of Covariance Matrix
    """

    #Constants
    ub_omega = 0.9
    lb_omega = 0.1
    ub_H0 = 100
    lb_H0 = 50
    range_omega = ub_omega - lb_omega
    range_H0 = ub_H0 - lb_H0


    def propose_new(H_0, omega_m, epsilon):
        H_0_new = H_0 + epsilon * range_H0 * np.random.randn()
        omega_m_new = omega_m + epsilon * range_omega * np.random.randn()
        return H_0_new, omega_m_new


    def log_likelihood(z, mu_obs, H_0, omega_m, inv_C):
        mu_th = theoretical_distance_modulus(z, H_0, omega_m)
        r = mu_obs - mu_th
        return -0.5 * r.T @ inv_C @ r


    def mh_step(omega_m, H_0, current_ll, mu_obs, z, inv_C):
        H_0_new, omega_m_new = propose_new(H_0, omega_m, epsilon)
        if not (lb_H0 < H_0_new < ub_H0 and lb_omega < omega_m_new < ub_omega):
            return omega_m, H_0, current_ll, False

        proposed_ll = log_likelihood(z, mu_obs, H_0_new, omega_m_new, inv_C)
        log_alpha = proposed_ll - current_ll
        if np.log(np.random.uniform()) < log_alpha:
            return omega_m_new, H_0_new, proposed_ll, True
        else:
            return omega_m, H_0, current_ll, False


    param_history = []
    H_0, omega_m = np.random.uniform(lb_H0, ub_H0), np.random.uniform(lb_omega, ub_omega)
    current_ll = log_likelihood(z, mu_obs, H_0, omega_m, inv_C)
    param_history.append([H_0, omega_m, current_ll, True])

    for s in range(iterations):
        if s % 1000 == 0:
            print(f"Iteration {s}")
        omega_m, H_0, ll, accept = mh_step(omega_m, H_0, current_ll, mu_obs, z, inv_C)
        current_ll = ll
        param_history.append([H_0, omega_m, ll, accept])
    param_history = np.array(param_history)

    return param_history