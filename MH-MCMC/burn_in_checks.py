import numpy as np

def estimate_burn_in(log_likelihood, window = 200, tol = 1e-2):
    for i in range(0, len(log_likelihood) - 2 * window):
        m1 = np.mean(log_likelihood[i:i+window])
        m2 = np.mean(log_likelihood[i+window:i+2*window])
        if abs(m2 - m1) < tol:
            return i + window
    print("Burn-in Not Found - Run the chain for longer")
    return None


def estimate_global_burnin(all_log_likelihoods):
    burn_ins = [estimate_burn_in(log_likelihood) for log_likelihood in all_log_likelihoods]
    burn_ins_clean = [b for b in burn_ins if b is not None]

    if len(burn_ins_clean) == 0:
        print("Warning: no chains converged — defaulting to halfway")
        default = len(all_log_likelihoods[0]) // 2
        return default, default

    print(f"Per-chain burn-ins: {burn_ins_clean}")
    return int(np.max(burn_ins_clean)), int(np.median(burn_ins_clean))
