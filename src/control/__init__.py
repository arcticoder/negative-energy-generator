"""
Real-Time Feedback Control Module
================================

Closed-loop control system for maintaining negative energy states.

This module provides:
1. State-space modeling of negative energy dynamics
2. H∞ robust control for disturbance rejection  
3. Model Predictive Control (MPC) for constraint handling
4. Hybrid control strategies combining H∞ and MPC
5. Actuator interface for boundary field control
6. Real-time implementation at GHz frequencies

Mathematical Foundation:
- State-space model: ẋ = Ax + Bu + w, y = Cx + v
- H∞ control: min_K ||T_{w→z}(K)||_∞ < γ
- MPC: finite-horizon QP with constraints
- Actuator dynamics: G(s) transfer functions

Control Objectives:
- Maintain ⟨T₀₀⟩ < 0 (negative energy density)
- Reject measurement noise and disturbances
- Respect actuator limits and safety constraints
- Optimize control effort and stability margins

Integration Points:
- Sensor input from hardware_instrumentation module
- Actuator output to boundary field systems
- Real-time execution loop at nanosecond timescales

Example Usage:
-------------
```python
from src.control import (
    StateSpaceModel, HInfinityController, ModelPredictiveController,
    RealTimeFeedbackController, run_closed_loop_simulation
)

# Create system model
system = StateSpaceModel(n_modes=4, n_actuators=3, n_sensors=2)

# Initialize controller  
controller = RealTimeFeedbackController(system, control_mode="hybrid")

# Control loop
for x_measured in sensor_data:
    u_command = controller.apply_control(x_measured)
    actuator_network.apply_command_vector(u_command, dt)
```
"""

from .real_time_feedback import (
    StateSpaceModel,
    HInfinityController, 
    ModelPredictiveController,
    RealTimeFeedbackController,
    run_closed_loop_simulation,
    demonstrate_control_strategies
)

__all__ = [
    'StateSpaceModel',
    'HInfinityController',
    'ModelPredictiveController', 
    'RealTimeFeedbackController',
    'run_closed_loop_simulation',
    'demonstrate_control_strategies'
]

__version__ = "1.0.0"
__author__ = "Negative Energy Control Systems Team"
