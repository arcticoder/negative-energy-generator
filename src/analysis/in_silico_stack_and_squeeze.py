# src/analysis/in_silico_stack_and_squeeze.py

import numpy as np
from typing import Dict

# ——— Multilayer Metamaterial Monte Carlo ———

def monte_carlo_multilayer(
    n_layers: int,
    eta: float = 0.95,
    beta: float = 0.5,
    sigma_d: float = 0.02,
    sigma_f: float = 0.02,
    n_samples: int = 5000
) -> Dict[str, float]:
    """
    Simulate N-layer stacking in-silico under Gaussian
    variation of layer thickness (d_k ~ N(1,σ_d²)) and
    filling fraction (f_k ~ N(f_nominal,σ_f²)), but purely
    computational—no fabrication step.

    g_k = η · d_k · f_k · k^(−β),   A = Σ_k g_k.

    Returns:
      mean_amp  : E[A]
      std_amp   : std(A)
      p_above   : P[A ≥ √N]  (fraction of runs beating √N baseline)
    """
    ks = np.arange(1, n_layers+1)
    target = np.sqrt(n_layers)
    amps = np.zeros(n_samples)
    for i in range(n_samples):
        d_k = np.random.normal(1.0, sigma_d, size=n_layers)
        f_k = np.clip(np.random.normal(1.0, sigma_f, size=n_layers), 0, 1)
        g_k = eta * d_k * f_k * ks**(-beta)
        amps[i] = g_k.sum()
    return {
        "mean_amp": float(np.mean(amps)),
        "std_amp":  float(np.std(amps)),
        "p_above_baseline": float((amps >= target).mean())
    }


# ——— JPA Monte Carlo over ε and Δ ———

def monte_carlo_jpa(
    Q: float,
    eps_nominal: float,
    delta_nominal: float = 0.0,
    sigma_eps: float = 0.01,
    sigma_delta: float = 0.01,
    n_samples: int = 5000
) -> Dict[str, float]:
    """
    Simulate in-silico jitter in pump amplitude ε and detuning Δ.
      r = ε·√(Q/10⁶) / (1 + 4Δ²)
      dB = 20·log10(e^r) = 8.686·r

    Returns:
      mean_db    : E[dB]
      std_db     : std(dB)
      p_above_15 : P[dB ≥ 15]
    """
    factor = np.sqrt(Q/1e6)
    dbs = np.zeros(n_samples)
    for i in range(n_samples):
        eps = np.random.normal(eps_nominal, sigma_eps)
        Δ   = np.random.normal(delta_nominal, sigma_delta)
        r   = eps * factor / (1 + 4*Δ**2)
        dbs[i] = 8.686 * r
    return {
        "mean_db": float(np.mean(dbs)),
        "std_db":  float(np.std(dbs)),
        "p_above_15dB": float((dbs >= 15.0).mean())
    }


# ——— In-silico Assessment Driver ———

if __name__ == "__main__":
    import pprint

    print("=== Multilayer Metamaterial In-Silico Assessment ===")
    ml_res = monte_carlo_multilayer(
        n_layers=10,
        eta=0.95,
        beta=0.5,
        sigma_d=0.02,
        sigma_f=0.02,
        n_samples=2000
    )
    pprint.pprint(ml_res)

    print("\n=== JPA Squeezing In-Silico Assessment ===")
    jpa_res = monte_carlo_jpa(
        Q=1e8,
        eps_nominal=0.2,
        delta_nominal=0.0,
        sigma_eps=0.01,
        sigma_delta=0.01,
        n_samples=2000
    )
    pprint.pprint(jpa_res)
