import numpy as np
import pandas as pd
import gzip
from pathlib import Path

def load_data():
    data_path = Path(__file__).parent / 'Pantheon+SH0ES.dat'
    cov_path = Path(__file__).parent / 'Pantheon+SH0ES_STAT+SYS.cov.gz'
    
    data = pd.read_csv(data_path, sep = ' ')
    z = data['zHD'].values
    mu_obs = data['MU_SH0ES'].values
    with gzip.open(cov_path, 'rt') as f:
        N = int(f.readline())
        C = np.fromstring(f.read(), sep = ' ').reshape(N, N)

    inv_C = np.linalg.inv(C)
    return z, mu_obs, inv_C
