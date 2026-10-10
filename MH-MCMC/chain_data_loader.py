import numpy as np

def load_chains(number_chains, chain_name):
    chains_df = []
    for i in range(number_chains):
        chains_df.append(np.loadtxt(f"{name}_chain_{i}.txt"))
    chains_df = np.array(chains_df)
    return chains_df
