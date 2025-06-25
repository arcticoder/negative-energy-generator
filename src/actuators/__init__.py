"""
Actuator Module Initialization
=============================

Interface layer for translating control commands into physical actuator signals.

This module provides:
1. Base actuator class with transfer function dynamics
2. Voltage modulators for boundary field control
3. Current drivers for magnetic coil systems
4. Laser modulators for optical field manipulation
5. Field shapers for electromagnetic focusing
6. Actuator network coordination and safety management

Actuator Types:
- VoltageModulator: High-voltage boundary field control (up to 1 MV)
- CurrentDriver: High-current magnetic coil drivers (up to 1 kA)
- LaserModulator: High-intensity laser control (up to 1 PW)
- FieldShaper: Electromagnetic field focusing (up to 100 TV/m)

Features:
- Transfer function modeling with bandwidth limitations
- Safety interlocks and protection systems
- Real-time command processing at GHz rates
- Inter-actuator coordination matrix
- Emergency shutdown capabilities

Example Usage:
-------------
```python
from src.actuators import ActuatorNetwork

# Create actuator network
network = ActuatorNetwork()

# Apply control commands
command_vector = np.array([1e5, 2e5, 100, 1e12, 1e13])  # V, V, A, W, V/m
outputs = network.apply_command_vector(command_vector, dt=1e-9)

# Monitor performance
status = network.get_network_status()
```
"""

from .boundary_field_actuators import (
    ActuatorBase,
    VoltageModulator,
    CurrentDriver, 
    LaserModulator,
    FieldShaper,
    ActuatorNetwork,
    demonstrate_actuator_system
)

__all__ = [
    'ActuatorBase',
    'VoltageModulator',
    'CurrentDriver',
    'LaserModulator', 
    'FieldShaper',
    'ActuatorNetwork',
    'demonstrate_actuator_system'
]

__version__ = "1.0.0"
__author__ = "Actuator Systems Team"
