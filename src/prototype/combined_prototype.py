import numpy as np
from casimir_array import optimize_casimir
from dynamic_casimir import sweep_dynamic
from squeezed_vacuum import optimize_single_mode
from metamaterial import optimize_meta

# 1 cm² area → ds as lengths, volumes normalized per m²
N = 5

ds_opt, E_array = optimize_casimir(N, 5e-9, 1e-8)
E_dyn, (d0,A,ω) = sweep_dynamic((5e-9,1e-8),(0,5e-9),(1e9,1e12))
r_opt, E_sq = optimize_single_mode(2*np.pi*1e14, 1e-12)
eps_opt, E_meta = optimize_meta(ds_opt, (1e-4,1))

print("Casimir array:", E_array)
print("Dynamic Casimir:", E_dyn)
print("Squeezed vac:", E_sq)
print("Metamaterial:", E_meta)
