import numpy as np
from quad_casimir import avg_casimir_gauss  # from earlier stub

def average_dynamic_energy(d0, A, ω):
    return avg_casimir_gauss(d0, A, ω, n=64)  # J/m³ averaged per second

def sweep_dynamic(d0_range, A_range, ω_range):
    best = (0,None)
    for d0 in np.linspace(*d0_range,10):
      for A in np.linspace(*A_range,10):
        for ω in np.linspace(*ω_range,10):
          E = average_dynamic_energy(d0, A, ω)
          if E < best[0]:
            best = (E, (d0,A,ω))
    return best
