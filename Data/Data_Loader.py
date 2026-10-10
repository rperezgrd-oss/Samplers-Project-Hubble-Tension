def load_data():
    data = pd.read_csv('Pantheon+SH0ES.dat', sep = ' ')
    z = data['zHD'].values
    mu_obs = data['MU_SH0ES'].values

    with open('Pantheon+SH0ES_STAT+SYS.cov.gz') as f:
        N = int(f.readline())
        C = np.array([float(x) for line in f for x in line.split()])
        C = C.reshape(N, N)

    inv_C = np.linalg.inv(C)
    return z, mu_obs, inv_C
