import numpy as np
from casimir_array import casimir_density

def meta_density(d, eps_eff):
    return casimir_density(d)/np.sqrt(eps_eff)

def optimize_meta(ds, eps_bounds):
    eps_min, eps_max = eps_bounds
    epss = np.linspace(eps_min, eps_max, 50)
    vals = [np.sum(meta_density(ds, e)*ds) for e in epss]
    i = np.argmin(vals)
    return epss[i], vals[i]  # best eps_eff and energy/m²
