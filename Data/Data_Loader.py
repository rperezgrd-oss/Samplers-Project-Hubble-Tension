import numpy as np
import pandas as pd
import gzip 

def load_data():
    data = pd.read_csv('Pantheon+SH0ES.dat', sep = ' ')
    z = data['zHD'].values
    mu_obs = data['MU_SH0ES'].values

    with gzip.open('Pantheon+SH0ES_STAT+SYS.cov.gz', 'rt') as f:
        N = int(f.readline())
        C = np.fromstring(f.read(), sep = ' ').reshape(N, N)

    inv_C = np.linalg.inv(C)
    return z, mu_obs, inv_C
