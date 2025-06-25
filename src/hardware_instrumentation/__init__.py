"""
Hardware Instrumentation Module
==============================

Precision measurement systems for negative energy density detection.

This module provides four key instrumentation classes:

1. InterferometricProbe - Phase-based ΔT₀₀ detection via electro-optic effects
2. CalorimetricSensor - Direct thermal measurement of energy density changes  
3. PhaseShiftInterferometer - Complete interferometry system with signal processing
4. RealTimeDAQ - High-speed data acquisition with triggering and buffering

Mathematical foundations:
- Interferometry: Δφ = (2π/λ) Δn L where Δn = ½n³rE, E = √(2μ₀|ΔT₀₀|)
- Calorimetry: ΔT = ΔE/(Cₚm) where ΔE = ΔT₀₀ × V
- Real-time sampling governed by Nyquist criterion: fs ≥ 2f_max

Example Usage:
-------------
```python
from src.hardware_instrumentation import InterferometricProbe, generate_T00_pulse

# Create probe
probe = InterferometricProbe(
    wavelength=1.55e-6,  # 1550 nm
    path_length=0.1,     # 10 cm
    n0=1.5,              # Glass
    r_coeff=1e-12        # m/V
)

# Generate test pulse
pulse = generate_T00_pulse("gaussian", -1e7, 1e-9, 0.2e-9)

# Measure
times = np.linspace(0, 5e-9, 1000)
T00_values = [pulse(t) for t in times]
result = probe.simulate_pulse(times, T00_values)

print(f"Max phase shift: {np.max(np.abs(result.values)):.2e} rad")
```
"""

from .diagnostics import (
    InterferometricProbe,
    CalorimetricSensor, 
    PhaseShiftInterferometer,
    RealTimeDAQ,
    MeasurementResult,
    generate_T00_pulse,
    benchmark_instrumentation_suite
)

__all__ = [
    'InterferometricProbe',
    'CalorimetricSensor',
    'PhaseShiftInterferometer', 
    'RealTimeDAQ',
    'MeasurementResult',
    'generate_T00_pulse',
    'benchmark_instrumentation_suite'
]

__version__ = "1.0.0"
__author__ = "Negative Energy Research Team"
__email__ = "research@negative-energy.dev"
