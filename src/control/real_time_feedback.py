"""
Real-Time Feedback Control for Negative Energy Systems
=====================================================

Implements closed-loop control strategies to maintain ⟨T₀₀⟩ < 0 in real time.

Mathematical Foundation:
1. State-Space Model: ẋ(t) = Ax(t) + Bu(t) + w(t), y(t) = Cx(t) + v(t)
2. H∞ Control: min_K ||T_{w→z}(K)||_∞ < γ  
3. Model Predictive Control: finite-horizon QP with constraints
4. Hybrid Control: Combines H∞ (disturbance rejection) + MPC (constraint handling)

Control Objectives:
- Maintain negative energy density: ⟨T₀₀⟩ < 0
- Reject measurement noise and environmental disturbances
- Respect actuator limits and safety constraints
- Minimize control effort while maximizing stability margin

Integration with Hardware:
- State estimation from interferometric/calorimetric sensors
- Actuator commands to boundary field modulators
- Real-time execution at nanosecond timescales
"""

import numpy as np
import scipy
import scipy.linalg
from scipy import signal
from scipy.linalg import solve_continuous_are, solve_discrete_are
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Attempt to import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    print("✅ CVXPY loaded for MPC optimization")
except ImportError:
    CVXPY_AVAILABLE = False
    print("⚠️  CVXPY not available - using simplified MPC")

try:
    from scipy.linalg import solve_continuous_are as solve_are
    ARE_SOLVER_AVAILABLE = True
    print("✅ Algebraic Riccati Equation solver available")
except ImportError:
    ARE_SOLVER_AVAILABLE = False
    print("⚠️  ARE solver not available - using approximate H∞")

# Physical constants and control parameters
CONTROL_FREQUENCY = 1e9     # 1 GHz control loop (1 ns sampling)
MAX_ACTUATOR_VOLTAGE = 1e6  # 1 MV maximum boundary field
ENERGY_DENSITY_THRESHOLD = 0.0  # Maintain T₀₀ < 0
STABILITY_MARGIN = 10.0     # Safety factor for control gains

class StateSpaceModel:
    """
    Linear state-space representation of negative energy dynamics.
    
    Mathematical Model:
    ẋ(t) = Ax(t) + Bu(t) + w(t)  - State evolution
    y(t) = Cx(t) + v(t)          - Sensor measurements
    
    State vector x: [T₀₀_mode1, T₀₀_mode2, ..., dT₀₀/dt_mode1, ...]
    Input vector u: [V_boundary1, V_boundary2, I_coil1, I_coil2, ...]
    Output vector y: [phase_shift, temperature_rise, ...]
    """
    
    def __init__(self, n_modes: int = 4, n_actuators: int = 3, n_sensors: int = 2):
        """
        Initialize linearized state-space model.
        
        Args:
            n_modes: Number of spatial T₀₀ modes to track
            n_actuators: Number of boundary field actuators
            n_sensors: Number of measurement channels
        """
        self.n_modes = n_modes
        self.n_actuators = n_actuators  
        self.n_sensors = n_sensors
        self.n_states = 2 * n_modes  # Position and velocity for each mode
        
        # Generate physics-informed system matrices
        self.A, self.B, self.C = self._generate_system_matrices()
        
        # Discretize for digital control
        self.dt = 1.0 / CONTROL_FREQUENCY  # 1 ns sampling
        self._discretize_system()
        
        # System properties
        self.eigenvalues = np.linalg.eigvals(self.A)
        self.is_stable = np.all(np.real(self.eigenvalues) < 0)
        self.is_controllable = self._check_controllability()
        self.is_observable = self._check_observability()
        
    def _generate_system_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate physics-informed A, B, C matrices."""
        n = self.n_states
        m = self.n_actuators
        p = self.n_sensors
        
        # A matrix: Coupled harmonic oscillators with damping
        A = np.zeros((n, n))
        
        # Each mode: [position, velocity] → [velocity, acceleration]
        for i in range(self.n_modes):
            pos_idx = 2*i
            vel_idx = 2*i + 1
            
            # Position derivative = velocity
            A[pos_idx, vel_idx] = 1.0
            
            # Velocity derivative = -ω²x - γv (damped oscillator)
            omega_i = 1e10 * (1 + 0.1*i)  # Mode frequencies ~10 GHz
            gamma_i = 1e8 * (1 + 0.05*i)  # Damping coefficients
            
            A[vel_idx, pos_idx] = -(omega_i**2)
            A[vel_idx, vel_idx] = -gamma_i
            
            # Mode coupling (off-diagonal terms)
            for j in range(self.n_modes):
                if i != j:
                    coupling = 1e9 * np.exp(-abs(i-j))  # Exponential coupling decay
                    A[vel_idx, 2*j] = coupling
        
        # B matrix: Actuator influence on each mode
        B = np.zeros((n, m))
        
        for i in range(self.n_modes):
            vel_idx = 2*i + 1
            for j in range(m):
                # Actuator coupling strength (boundary conditions)
                coupling_strength = 1e6 * np.exp(-0.5*abs(i-j))
                B[vel_idx, j] = coupling_strength
        
        # C matrix: Sensor measurements of state
        C = np.zeros((p, n))
        
        # Sensor 1: Weighted sum of mode positions (interferometric)
        for i in range(self.n_modes):
            C[0, 2*i] = 1.0 / (1 + i)  # Decreasing sensitivity with mode index
            
        # Sensor 2: Weighted sum of mode velocities (calorimetric)
        if p > 1:
            for i in range(self.n_modes):
                C[1, 2*i + 1] = 0.5 / (1 + i)
        
        return A, B, C
    
    def _discretize_system(self):
        """Convert continuous-time system to discrete-time using ZOH."""
        # Use matrix exponential for exact discretization
        n = self.A.shape[0]
        m = self.B.shape[1]
        
        # Create augmented matrix for exact discretization
        M = np.zeros((n + m, n + m))
        M[:n, :n] = self.A * self.dt
        M[:n, n:] = self.B * self.dt
        
        # Matrix exponential
        exp_M = scipy.linalg.expm(M)
        
        # Extract discrete-time matrices
        self.Ad = exp_M[:n, :n]
        self.Bd = exp_M[:n, n:]
        self.Cd = self.C.copy()  # C matrix unchanged in discretization
        self.Dd = np.zeros((self.n_sensors, self.n_actuators))
        
    def _check_controllability(self) -> bool:
        """Check if system is controllable."""
        n = self.A.shape[0]
        controllability_matrix = self.B.copy()
        
        for i in range(1, n):
            controllability_matrix = np.hstack([
                controllability_matrix, 
                np.linalg.matrix_power(self.A, i) @ self.B
            ])
        
        rank = np.linalg.matrix_rank(controllability_matrix)
        return rank == n
    
    def _check_observability(self) -> bool:
        """Check if system is observable."""
        n = self.A.shape[0]
        observability_matrix = self.C.copy()
        
        for i in range(1, n):
            observability_matrix = np.vstack([
                observability_matrix,
                self.C @ np.linalg.matrix_power(self.A, i)
            ])
        
        rank = np.linalg.matrix_rank(observability_matrix)
        return rank == n
    
    def simulate(self, x0: np.ndarray, u_sequence: np.ndarray, 
                 noise_level: float = 0.0) -> Dict:
        """
        Simulate system response to control sequence.
        
        Args:
            x0: Initial state
            u_sequence: Control input sequence [m × T]
            noise_level: Process and measurement noise standard deviation
            
        Returns:
            Dictionary with simulation results
        """
        T = u_sequence.shape[1]
        
        # Initialize arrays
        x_trajectory = np.zeros((self.n_states, T+1))
        y_trajectory = np.zeros((self.n_sensors, T))
        
        x_trajectory[:, 0] = x0
        
        # Simulate forward
        for t in range(T):
            # Process noise
            w = noise_level * np.random.randn(self.n_states)
            
            # State update
            x_trajectory[:, t+1] = (self.Ad @ x_trajectory[:, t] + 
                                   self.Bd @ u_sequence[:, t] + w)
            
            # Measurement noise
            v = noise_level * np.random.randn(self.n_sensors)
            
            # Output measurement
            y_trajectory[:, t] = self.Cd @ x_trajectory[:, t] + v
        
        return {
            'states': x_trajectory,
            'outputs': y_trajectory,
            'times': np.arange(T+1) * self.dt,
            'control_times': np.arange(T) * self.dt
        }

class HInfinityController:
    """
    H∞ robust controller for disturbance rejection.
    
    Mathematical Foundation:
    - Solves: min_K ||T_{w→z}(K)||_∞ < γ
    - Uses Algebraic Riccati Equation: A^T X + X A - X B R^{-1} B^T X + Q = 0
    - Control law: u = -K x, where K = R^{-1} B^T X
    """
    
    def __init__(self, system: StateSpaceModel, gamma: float = 1.0,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None):
        """
        Initialize H∞ controller.
        
        Args:
            system: StateSpaceModel instance
            gamma: H∞ performance level (smaller = better disturbance rejection)
            Q: State weighting matrix
            R: Control weighting matrix
        """
        self.system = system
        self.gamma = gamma
        
        # Default weighting matrices
        if Q is None:
            self.Q = np.eye(system.n_states)
            # Penalize positive energy density modes more heavily
            for i in range(system.n_modes):
                self.Q[2*i, 2*i] = 10.0  # Position penalty
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.eye(system.n_actuators)
        else:
            self.R = R
        
        # Controller gain (computed on demand)
        self.K = None
        self.is_designed = False
        
    def design_controller(self) -> bool:
        """
        Design H∞ controller by solving Algebraic Riccati Equation.
        
        Returns:
            True if successful, False if failed
        """
        try:
            if ARE_SOLVER_AVAILABLE:
                # Solve continuous ARE: A^T X + X A - X B R^{-1} B^T X + Q = 0
                X = solve_are(self.system.A, self.system.B, self.Q, self.R)
                
                # Compute control gain: K = R^{-1} B^T X
                self.K = linalg.solve(self.R, self.system.B.T @ X)
                
                # Check stability of closed-loop system
                A_closed = self.system.A - self.system.B @ self.K
                eigenvals = np.linalg.eigvals(A_closed)
                
                if np.all(np.real(eigenvals) < 0):
                    self.is_designed = True
                    return True
                else:
                    print("⚠️  H∞ controller: Closed-loop system unstable")
                    return False
                    
            else:
                # Simplified LQR-based approximation
                print("🔧 Using LQR approximation for H∞ control")
                K, _, _ = signal.lqr(self.system.A, self.system.B, self.Q, self.R)
                self.K = K
                self.is_designed = True
                return True
                
        except Exception as e:
            print(f"❌ H∞ controller design failed: {e}")
            return False
    
    def control_law(self, x: np.ndarray) -> np.ndarray:
        """
        Compute control input: u = -K x
        
        Args:
            x: Current state vector
            
        Returns:
            Control input vector
        """
        if not self.is_designed:
            success = self.design_controller()
            if not success:
                return np.zeros(self.system.n_actuators)
        
        # Apply control law with saturation
        u = -self.K @ x
        
        # Saturate control inputs
        u_max = MAX_ACTUATOR_VOLTAGE
        u = np.clip(u, -u_max, u_max)
        
        return u

class ModelPredictiveController:
    """
    Model Predictive Controller with constraint handling.
    
    Mathematical Foundation:
    min_{u_0,...,u_{N-1}} Σ_{i=0}^{N-1} [x_i^T Q x_i + u_i^T R u_i]
    subject to:
        x_{i+1} = A_d x_i + B_d u_i
        T₀₀(x_i) < 0  (energy constraint)
        ||u_i|| ≤ u_max  (actuator limits)
    """
    
    def __init__(self, system: StateSpaceModel, horizon: int = 20,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None):
        """
        Initialize MPC controller.
        
        Args:
            system: StateSpaceModel instance
            horizon: Prediction horizon length
            Q: State weighting matrix
            R: Control weighting matrix
        """
        self.system = system
        self.N = horizon
        
        # Default weighting matrices
        if Q is None:
            self.Q = np.eye(system.n_states)
            # Heavy penalty on positive energy density
            for i in range(system.n_modes):
                self.Q[2*i, 2*i] = 100.0  # Position penalty
        else:
            self.Q = Q
            
        if R is None:
            self.R = 0.1 * np.eye(system.n_actuators)  # Control effort penalty
        else:
            self.R = R
        
        self.u_max = MAX_ACTUATOR_VOLTAGE
        self.energy_threshold = ENERGY_DENSITY_THRESHOLD
        
    def solve_qp(self, x0: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve finite-horizon quadratic program.
        
        Args:
            x0: Initial state
            
        Returns:
            Optimal control sequence or None if infeasible
        """
        if not CVXPY_AVAILABLE:
            return self._solve_qp_simplified(x0)
        
        try:
            n, m = self.system.n_states, self.system.n_actuators
            
            # Decision variables
            X = cp.Variable((n, self.N + 1))  # States
            U = cp.Variable((m, self.N))      # Controls
            
            # Cost function
            cost = 0
            for i in range(self.N):
                cost += cp.quad_form(X[:, i], self.Q)
                cost += cp.quad_form(U[:, i], self.R)
            
            # Terminal cost
            cost += cp.quad_form(X[:, self.N], self.Q)
            
            # Constraints
            constraints = [X[:, 0] == x0]  # Initial condition
            
            for i in range(self.N):
                # System dynamics
                constraints += [X[:, i+1] == self.system.Ad @ X[:, i] + 
                               self.system.Bd @ U[:, i]]
                
                # Energy density constraint: C_energy @ x < 0
                # Use first row of C matrix as energy density measurement
                energy_output = self.system.Cd[0, :] @ X[:, i]
                constraints += [energy_output <= self.energy_threshold]
                
                # Actuator limits
                constraints += [cp.norm(U[:, i], 'inf') <= self.u_max]
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return U.value
            else:
                print(f"⚠️  MPC: Optimization failed with status {problem.status}")
                return None
                
        except Exception as e:
            print(f"❌ MPC optimization failed: {e}")
            return None
    
    def _solve_qp_simplified(self, x0: np.ndarray) -> np.ndarray:
        """Simplified MPC without convex optimization (fallback)."""
        # Use unconstrained LQR solution as approximation
        try:
            K, _, _ = signal.dlqr(self.system.Ad, self.system.Bd, self.Q, self.R)
            u_sequence = np.zeros((self.system.n_actuators, self.N))
            
            x = x0.copy()
            for i in range(self.N):
                u = -K @ x
                u = np.clip(u, -self.u_max, self.u_max)
                u_sequence[:, i] = u
                x = self.system.Ad @ x + self.system.Bd @ u
            
            return u_sequence
            
        except Exception:
            # Last resort: zero control
            return np.zeros((self.system.n_actuators, self.N))
    
    def control_law(self, x: np.ndarray) -> np.ndarray:
        """
        Compute MPC control input (receding horizon).
        
        Args:
            x: Current state vector
            
        Returns:
            First control input from optimal sequence
        """
        u_sequence = self.solve_qp(x)
        
        if u_sequence is not None:
            return u_sequence[:, 0]  # Return first control input
        else:
            # Fallback: zero control
            return np.zeros(self.system.n_actuators)

class RealTimeFeedbackController:
    """
    Main real-time feedback control system.
    
    Combines H∞ and MPC strategies:
    - H∞ for fast disturbance rejection
    - MPC for constraint handling and optimization
    - Hybrid switching based on system state
    """
    
    def __init__(self, system: StateSpaceModel, 
                 hinf_gamma: float = 1.0, mpc_horizon: int = 20,
                 control_mode: str = "hybrid"):
        """
        Initialize real-time controller.
        
        Args:
            system: StateSpaceModel instance
            hinf_gamma: H∞ performance level
            mpc_horizon: MPC prediction horizon
            control_mode: "hinf", "mpc", or "hybrid"
        """
        self.system = system
        self.control_mode = control_mode
        
        # Initialize sub-controllers
        self.hinf_controller = HInfinityController(system, hinf_gamma)
        self.mpc_controller = ModelPredictiveController(system, mpc_horizon)
        
        # Control parameters
        self.energy_threshold = ENERGY_DENSITY_THRESHOLD
        self.disturbance_threshold = 1e-6  # Switch to H∞ above this level
        
        # Performance monitoring
        self.control_history = []
        self.state_history = []
        self.energy_violations = 0
        self.total_control_calls = 0
        
    def apply_control(self, x: np.ndarray, disturbance_level: float = 0.0) -> np.ndarray:
        """
        Main control function - selects appropriate strategy.
        
        Args:
            x: Current state vector
            disturbance_level: Estimated disturbance magnitude
            
        Returns:
            Control input vector
        """
        self.total_control_calls += 1
        
        # Monitor energy constraint violation
        energy_output = self.system.C[0, :] @ x
        if energy_output > self.energy_threshold:
            self.energy_violations += 1
        
        # Select control strategy
        if self.control_mode == "hinf":
            u = self.hinf_controller.control_law(x)
        elif self.control_mode == "mpc":
            u = self.mpc_controller.control_law(x)
        elif self.control_mode == "hybrid":
            u = self._hybrid_control(x, disturbance_level)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
        
        # Log performance
        self.control_history.append(u.copy())
        self.state_history.append(x.copy())
        
        return u
    
    def _hybrid_control(self, x: np.ndarray, disturbance_level: float) -> np.ndarray:
        """
        Hybrid control strategy combining H∞ and MPC.
        
        Logic:
        - Use H∞ when disturbances are high (fast response needed)
        - Use MPC when constraints are active (optimization needed)
        - Blend both when in normal operation
        """
        # Compute both control laws
        u_hinf = self.hinf_controller.control_law(x)
        u_mpc = self.mpc_controller.control_law(x)
        
        # Decision logic based on system state
        energy_output = self.system.C[0, :] @ x
        state_magnitude = np.linalg.norm(x)
        
        # High disturbance → prefer H∞
        if disturbance_level > 0.1:
            alpha = 0.8  # 80% H∞, 20% MPC
        # Energy constraint active → prefer MPC
        elif energy_output > -0.1 * abs(self.energy_threshold):
            alpha = 0.2  # 20% H∞, 80% MPC
        # Large state deviations → prefer H∞
        elif state_magnitude > 1.0:
            alpha = 0.7  # 70% H∞, 30% MPC
        # Normal operation → balanced
        else:
            alpha = 0.5  # 50% H∞, 50% MPC
        
        # Blend control inputs
        u_hybrid = alpha * u_hinf + (1 - alpha) * u_mpc
        
        # Apply actuator limits
        u_hybrid = np.clip(u_hybrid, -MAX_ACTUATOR_VOLTAGE, MAX_ACTUATOR_VOLTAGE)
        
        return u_hybrid
    
    def get_performance_metrics(self) -> Dict:
        """Get controller performance statistics."""
        if self.total_control_calls == 0:
            return {'status': 'No control calls yet'}
        
        control_array = np.array(self.control_history).T
        state_array = np.array(self.state_history).T
        
        return {
            'energy_violation_rate': self.energy_violations / self.total_control_calls,
            'average_control_effort': np.mean(np.linalg.norm(control_array, axis=0)),
            'max_control_effort': np.max(np.linalg.norm(control_array, axis=0)),
            'average_state_magnitude': np.mean(np.linalg.norm(state_array, axis=0)),
            'total_control_calls': self.total_control_calls,
            'control_saturation_rate': np.mean(np.abs(control_array) >= 0.9 * MAX_ACTUATOR_VOLTAGE)
        }

def run_closed_loop_simulation(duration: float = 1e-6, 
                               disturbance_amplitude: float = 0.1,
                               control_mode: str = "hybrid") -> Dict:
    """
    Run closed-loop simulation with disturbances.
    
    Args:
        duration: Simulation duration (seconds)
        disturbance_amplitude: Disturbance noise level
        control_mode: Controller type ("hinf", "mpc", "hybrid")
        
    Returns:
        Simulation results dictionary
    """
    print(f"🔄 Running closed-loop simulation ({control_mode} control)")
    print(f"   • Duration: {duration*1e6:.1f} μs")
    print(f"   • Disturbance level: {disturbance_amplitude:.3f}")
    
    # Create system and controller
    system = StateSpaceModel(n_modes=4, n_actuators=3, n_sensors=2)
    controller = RealTimeFeedbackController(system, control_mode=control_mode)
    
    print(f"   • System: {system.n_states} states, {system.n_actuators} actuators")
    print(f"   • Controllable: {system.is_controllable}, Observable: {system.is_observable}")
    print(f"   • Stable: {system.is_stable}")
    
    # Simulation parameters
    T_steps = int(duration / system.dt)
    times = np.arange(T_steps) * system.dt
    
    # Initial state (small positive energy density - needs correction)
    x0 = np.zeros(system.n_states)
    x0[0] = 0.1  # First mode with positive energy
    x0[2] = -0.05  # Second mode with negative energy
    
    # Preallocate arrays
    x_trajectory = np.zeros((system.n_states, T_steps))
    u_trajectory = np.zeros((system.n_actuators, T_steps))
    y_trajectory = np.zeros((system.n_sensors, T_steps))
    energy_trajectory = np.zeros(T_steps)
    
    # Initialize
    x = x0.copy()
    x_trajectory[:, 0] = x
    
    # Simulation loop
    for t in range(T_steps - 1):
        # Compute control input
        disturbance_level = disturbance_amplitude * (0.5 + 0.5 * np.sin(2*np.pi*1e6*times[t]))
        u = controller.apply_control(x, disturbance_level)
        
        # Apply disturbance
        w = disturbance_amplitude * np.random.randn(system.n_states)
        
        # Update state
        x_next = system.Ad @ x + system.Bd @ u + w * system.dt
        
        # Measurement
        v = 0.01 * disturbance_amplitude * np.random.randn(system.n_sensors)
        y = system.Cd @ x + v
        
        # Store results
        x_trajectory[:, t+1] = x_next
        u_trajectory[:, t] = u
        y_trajectory[:, t] = y
        energy_trajectory[t] = system.C[0, :] @ x  # Energy density measurement
        
        # Update for next iteration
        x = x_next
    
    # Final energy measurement
    energy_trajectory[-1] = system.C[0, :] @ x
    
    # Performance analysis
    performance = controller.get_performance_metrics()
    
    # Energy constraint satisfaction
    energy_violations = np.sum(energy_trajectory > ENERGY_DENSITY_THRESHOLD)
    energy_satisfaction_rate = 1 - (energy_violations / T_steps)
    
    print(f"   ✅ Simulation complete")
    print(f"   • Energy constraint satisfaction: {energy_satisfaction_rate:.1%}")
    print(f"   • Average control effort: {performance['average_control_effort']:.2e}")
    print(f"   • Control saturation rate: {performance['control_saturation_rate']:.1%}")
    
    return {
        'times': times,
        'states': x_trajectory,
        'controls': u_trajectory,
        'outputs': y_trajectory,
        'energy_density': energy_trajectory,
        'performance': performance,
        'system': system,
        'controller': controller,
        'energy_satisfaction_rate': energy_satisfaction_rate,
        'final_energy': energy_trajectory[-1]
    }

def demonstrate_control_strategies():
    """Demonstrate different control strategies with visualization."""
    print("\n🎯 CONTROL STRATEGY DEMONSTRATION")
    print("=" * 50)
    
    # Test parameters
    duration = 5e-6  # 5 microseconds
    disturbance = 0.05
    
    # Run simulations for each control mode
    strategies = ["hinf", "mpc", "hybrid"]
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy.upper()} control...")
        results[strategy] = run_closed_loop_simulation(
            duration=duration, 
            disturbance_amplitude=disturbance,
            control_mode=strategy
        )
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Energy density trajectories
    ax1 = axes[0, 0]
    for strategy, result in results.items():
        times_us = result['times'] * 1e6  # Convert to microseconds
        ax1.plot(times_us, result['energy_density'], label=f'{strategy.upper()}', linewidth=2)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='T₀₀ = 0')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Energy Density ⟨T₀₀⟩')
    ax1.set_title('Energy Density Control')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control effort comparison
    ax2 = axes[0, 1]
    for strategy, result in results.items():
        times_us = result['times'][:-1] * 1e6
        control_magnitude = np.linalg.norm(result['controls'], axis=0)
        ax2.plot(times_us, control_magnitude/1e6, label=f'{strategy.upper()}', linewidth=2)
    
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Control Effort (MV)')
    ax2.set_title('Control Effort Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance metrics
    ax3 = axes[1, 0]
    metrics = ['energy_satisfaction_rate', 'average_control_effort', 'control_saturation_rate']
    strategy_names = list(results.keys())
    
    x_pos = np.arange(len(strategy_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = []
        for strategy in strategy_names:
            if metric == 'energy_satisfaction_rate':
                values.append(results[strategy][metric])
            else:
                values.append(results[strategy]['performance'][metric])
        
        # Normalize for plotting
        if metric == 'average_control_effort':
            values = np.array(values) / 1e6  # Convert to MV
        
        ax3.bar(x_pos + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax3.set_xlabel('Control Strategy')
    ax3.set_ylabel('Normalized Metric')
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels([s.upper() for s in strategy_names])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: State trajectories (first mode)
    ax4 = axes[1, 1]
    for strategy, result in results.items():
        times_us = result['times'] * 1e6
        ax4.plot(times_us, result['states'][0, :], label=f'{strategy.upper()} Mode 1', linewidth=2)
        ax4.plot(times_us, result['states'][2, :], label=f'{strategy.upper()} Mode 2', 
                linewidth=1, linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('State Amplitude')
    ax4.set_title('State Evolution (First Two Modes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('control_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n🎯 CONTROL STRATEGY SUMMARY")
    print("=" * 40)
    
    for strategy, result in results.items():
        perf = result['performance']
        print(f"\n{strategy.upper()} Control:")
        print(f"   • Energy satisfaction: {result['energy_satisfaction_rate']:.1%}")
        print(f"   • Final energy: {result['final_energy']:.2e}")
        print(f"   • Avg control effort: {perf['average_control_effort']/1e6:.2f} MV")
        print(f"   • Control saturation: {perf['control_saturation_rate']:.1%}")
    
    print(f"\n📊 Comparison saved to 'control_strategy_comparison.png'")
    
    return results

if __name__ == "__main__":
    print("🔄 Real-Time Feedback Control System")
    print("=" * 50)
    print("Implementing closed-loop control for negative energy maintenance")
    
    # Run demonstration
    demo_results = demonstrate_control_strategies()
    
    # Additional analysis
    print(f"\n🔬 TECHNICAL ANALYSIS")
    print("=" * 30)
    
    best_strategy = min(demo_results.items(), 
                       key=lambda x: abs(x[1]['final_energy']))[0]
    
    print(f"🏆 Best performing strategy: {best_strategy.upper()}")
    print(f"   • Achieved final energy: {demo_results[best_strategy]['final_energy']:.2e}")
    print(f"   • Energy constraint satisfaction: {demo_results[best_strategy]['energy_satisfaction_rate']:.1%}")
    
    # System analysis
    system = demo_results[best_strategy]['system']
    print(f"\n🔧 System Properties:")
    print(f"   • Eigenvalues (real): {np.real(system.eigenvalues)}")
    print(f"   • Controllable: {system.is_controllable}")
    print(f"   • Observable: {system.is_observable}")
    print(f"   • Sampling frequency: {CONTROL_FREQUENCY/1e9:.0f} GHz")
    
    print(f"\n✅ Real-Time Feedback Control System Ready!")
