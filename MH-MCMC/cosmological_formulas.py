import numpy as np
from scipy.integrate import quad

def expansion_rate(z, omega_m):

    #The Expansion Rate is defined according to the LCDM Model as follows:

    #E(z) = \sqrt{\Omega_m(1 + z)^3 +(1 - \Omega_m)}

    #- omega_m: Matter Density Parameter
    #- z: Redshift

    return np.sqrt(omega_m * ((1 + z)**3 - 1) + 1)


def luminosity_distance(z, H0, omega_m, c = 299792458):

    #The Luminosity Distance is defined according to the LCDM Model as follows:

    #dL(z) = \frac{c(1 + z')}{H0} \int^{z'}_{0} \frac{1}{E(z')}dz'

    #- H0: Hubble Constant
    #- z': Redshift
    #- c: Speed of Light

    dL_z = np.ones(len(z))
    coefficient = (c / H0) * (1 + np.array(z))
    for i in range(len(z)):
        dL_z[i] = quad(lambda z: 1 / expansion_rate(z, omega_m), 0, z[i])[0]

    return dL_z * coefficient


def theoretical_distance_modulus(z, H0, omega_m):

    #The Luminosity Distance is defined according to the LCDM Model as follows:

    #dL(z) = \frac{c(1 + z')}{H0} \int^{z'}_{0} \frac{1}{E(z')}dz'

    #- H0: Hubble Constant
    #- z': Redshift
    #- c: Speed of Light
        
    return 5 * np.log10(luminosity_distance(z, H0, omega_m)) + 25


"""    def simpson_int(z, omega_m, function):
        N = max(int(np.ceil(z / (0.001 * z))), 11)
        if N % 2 == 0:
            N += 1
        grid_z = np.linspace(0, z, N)
        delta_z = grid_z[1] - grid_z[0]
        integrand = function(grid_z, omega_m)
        return delta_z / 3 * (integrand[0] + integrand[-1] + 4 * np.sum(integrand[1:-1:2]) + 2 * np.sum(integrand[2:-2:2]))"""
