import numpy as np
hbar, c = 1.054e-34, 3e8

def casimir_density(d):
    return - (np.pi**2 * hbar * c) / (720 * d**4)

def total_array_energy(ds):
    # energy per unit area (J/m²)
    return np.sum(casimir_density(ds) * ds)

from scipy.optimize import minimize
def optimize_casimir(N, d_min, d_max):
    x0 = np.ones(N) * (d_min + d_max)/2
    bounds = [(d_min, d_max)]*N
    res = minimize(lambda d: -total_array_energy(d), x0, bounds=bounds)
    return res.x, total_array_energy(res.x)
