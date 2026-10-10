import numpy as np

def load_chains(number_chains, chain_name):
    chains_df = []
    for i in range(number_chains):
        chain = np.loadtxt(f"{chain_name}_chain_{i}.txt")
        chains_df.append(chain)
    chains_df = np.asarray(chains_df)
    return chains_df
