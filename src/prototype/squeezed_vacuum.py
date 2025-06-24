import numpy as np
hbar = 1.054e-34
def squeezed_density(omega, r, V):
    return - (hbar*omega)/(2*V) * np.sinh(2*r)

from scipy.optimize import minimize_scalar
def optimize_single_mode(omega, V, r_max=3.0):
    f = lambda r: -squeezed_density(omega, r, V)  # want most negative
    res = minimize_scalar(f, bounds=(0, r_max), method='bounded')
    return res.x, squeezed_density(omega, res.x, V)
