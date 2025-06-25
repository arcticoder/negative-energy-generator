"""
Physics-Informed Neural Network (PINN) for Direct T₀₀ Optimization
================================================================

This module implements neural network-based optimization where the network
parameterizes warp-bubble profiles or field configurations and is trained
to directly maximize negative energy through automatic differentiation.

Mathematical Foundation:
    NN(·|θ) outputs profile f(r,t;θ) on grid
    Loss: L(θ) = ∫ T₀₀(f(r,t;θ)) d³x
    Update: θ ← θ - η ∇_θ L
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Callable, Optional, Any
import json
from datetime import datetime
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.autograd import grad
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class ProfileNet(nn.Module):
    """
    Neural network for parameterizing exotic matter field profiles.
    
    Can represent warp bubble profiles, field configurations, or
    any spatially-varying exotic matter distribution.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, 
                 n_layers: int = 4, output_dim: int = 1, activation: str = "tanh"):
        """
        Initialize the profile network.
        
        Args:
            input_dim: Input dimensionality (3 for spatial coordinates)
            hidden_dim: Hidden layer width
            n_layers: Number of hidden layers
            output_dim: Output dimensionality (1 for scalar field)
            activation: Activation function ("tanh", "relu", "swish")
        """
        super(ProfileNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        
        # Choose activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # Swish/SiLU
        else:
            self.activation = nn.Tanh()
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Final activation to ensure proper range
        if output_dim == 1:
            layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input coordinates [batch_size, input_dim]
            
        Returns:
            Field values [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives of the field with respect to input coordinates.
        
        Args:
            x: Input coordinates [batch_size, 3] (x, y, z)
            
        Returns:
            Dictionary with gradient components
        """
        x.requires_grad_(True)
        f = self.forward(x)
        
        # Compute gradients
        f_x = grad(f, x, grad_outputs=torch.ones_like(f), 
                  create_graph=True, retain_graph=True)[0]
        
        derivatives = {
            'f': f,
            'f_x': f_x[:, 0:1] if x.shape[1] > 0 else None,
            'f_y': f_x[:, 1:2] if x.shape[1] > 1 else None,
            'f_z': f_x[:, 2:3] if x.shape[1] > 2 else None,
            'f_grad': f_x
        }
        
        # Second derivatives (Hessian diagonal)
        if f_x is not None:
            f_xx = grad(f_x[:, 0], x, grad_outputs=torch.ones_like(f_x[:, 0]),
                       create_graph=True, retain_graph=True)[0][:, 0:1] if x.shape[1] > 0 else None
            f_yy = grad(f_x[:, 1], x, grad_outputs=torch.ones_like(f_x[:, 1]),
                       create_graph=True, retain_graph=True)[0][:, 1:2] if x.shape[1] > 1 else None
            f_zz = grad(f_x[:, 2], x, grad_outputs=torch.ones_like(f_x[:, 2]),
                       create_graph=True, retain_graph=True)[0][:, 2:3] if x.shape[1] > 2 else None
            
            derivatives.update({
                'f_xx': f_xx,
                'f_yy': f_yy,
                'f_zz': f_zz,
                'laplacian': (f_xx + f_yy + f_zz) if all(d is not None for d in [f_xx, f_yy, f_zz]) else None
            })
        
        return derivatives


class ExoticMatterPINN:
    """
    Physics-Informed Neural Network for exotic matter optimization.
    
    Trains a neural network to parameterize field configurations that
    maximize negative energy density under physical constraints.
    """
    
    def __init__(self, network: ProfileNet, energy_computer: Callable,
                 device: str = "auto", learning_rate: float = 1e-3):
        """
        Initialize the PINN optimizer.
        
        Args:
            network: Neural network for field parameterization
            energy_computer: Function to compute T₀₀ from field values and derivatives
            device: Device for computation ("cpu", "cuda", or "auto")
            learning_rate: Learning rate for optimization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PINN optimization")
        
        self.network = network
        self.energy_computer = energy_computer
        self.learning_rate = learning_rate
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.network.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = []
        self.best_loss = np.inf
        self.best_state = None
    
    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-based loss function.
        
        Args:
            x: Coordinate grid [batch_size, 3]
            
        Returns:
            Physics loss (negative for energy minimization)
        """
        # Get field and derivatives
        derivs = self.network.get_derivatives(x)
        
        # Compute T₀₀ using the energy computer
        T00 = self.energy_computer(derivs, x)
        
        # Energy integral (approximate with batch mean)
        energy_density = T00.mean()
        
        # We want to maximize negative energy, so minimize positive energy
        # Return energy as loss (lower energy = lower loss)
        return energy_density
    
    def compute_constraint_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute physics constraint penalties.
        
        Args:
            x: Coordinate grid
            
        Returns:
            Constraint penalty term
        """
        derivs = self.network.get_derivatives(x)
        f = derivs['f']
        
        # Example constraints (modify based on your physics)
        constraint_loss = 0.0
        
        # Boundary conditions (field should go to zero at boundaries)
        boundary_mask = (torch.abs(x).max(dim=1)[0] > 0.8)  # Points near boundary
        if boundary_mask.any():
            boundary_penalty = torch.mean(f[boundary_mask]**2)
            constraint_loss += boundary_penalty * 0.1
        
        # Smoothness constraint (penalize large gradients)
        if derivs['f_grad'] is not None:
            smoothness_penalty = torch.mean(derivs['f_grad']**2)
            constraint_loss += smoothness_penalty * 0.01
        
        # Energy scale constraint (prevent runaway solutions)
        energy_scale_penalty = torch.mean(f**4)  # Quartic penalty
        constraint_loss += energy_scale_penalty * 0.001
        
        return constraint_loss
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            x: Coordinate batch
            
        Returns:
            Dictionary with loss components
        """
        self.optimizer.zero_grad()
        
        # Compute loss components
        physics_loss = self.compute_physics_loss(x)
        constraint_loss = self.compute_constraint_loss(x)
        
        # Total loss
        total_loss = physics_loss + constraint_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Return loss components
        return {
            'total_loss': total_loss.item(),
            'physics_loss': physics_loss.item(),
            'constraint_loss': constraint_loss.item()
        }
    
    def train(self, n_epochs: int = 1000, batch_size: int = 1000,
              domain_bounds: Tuple[float, float] = (-1.0, 1.0),
              verbose: bool = True, save_frequency: int = 100) -> Dict:
        """
        Train the PINN to optimize exotic matter configuration.
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size for coordinate sampling
            domain_bounds: Spatial domain bounds (min, max)
            verbose: Whether to print progress
            save_frequency: How often to save best model
            
        Returns:
            Training results dictionary
        """
        if verbose:
            print(f"🧠 Starting PINN Training for Exotic Matter Optimization")
            print(f"   • Network parameters: {sum(p.numel() for p in self.network.parameters())}")
            print(f"   • Device: {self.device}")
            print(f"   • Epochs: {n_epochs}")
            print(f"   • Batch size: {batch_size}")
            print(f"   • Domain: [{domain_bounds[0]}, {domain_bounds[1]}]³")
        
        min_bound, max_bound = domain_bounds
        
        # Training loop
        for epoch in range(n_epochs):
            # Sample random coordinates in domain
            x = torch.rand(batch_size, 3, device=self.device)
            x = x * (max_bound - min_bound) + min_bound
            
            # Training step
            losses = self.train_step(x)
            
            # Track best model
            if losses['total_loss'] < self.best_loss:
                self.best_loss = losses['total_loss']
                self.best_state = self.network.state_dict().copy()
            
            # Store history
            epoch_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **losses
            }
            self.training_history.append(epoch_data)
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"   Epoch {epoch:4d}: total_loss={losses['total_loss']:.3e}, "
                      f"physics={losses['physics_loss']:.3e}, "
                      f"constraint={losses['constraint_loss']:.3e}")
        
        # Load best model
        if self.best_state is not None:
            self.network.load_state_dict(self.best_state)
        
        training_result = {
            'success': True,
            'n_epochs': n_epochs,
            'best_loss': self.best_loss,
            'final_loss': losses['total_loss'],
            'training_history': self.training_history,
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'device_used': str(self.device)
        }
        
        if verbose:
            print(f"✅ Training completed!")
            print(f"   • Best loss: {self.best_loss:.3e}")
            print(f"   • Final loss: {losses['total_loss']:.3e}")
            print(f"   • Improvement: {(self.training_history[0]['total_loss'] - self.best_loss):.3e}")
        
        return training_result
    
    def evaluate_field(self, grid_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate the optimized field on a grid of points.
        
        Args:
            grid_points: Grid coordinates [N, 3]
            
        Returns:
            Dictionary with field values and properties
        """
        self.network.eval()
        
        with torch.no_grad():
            derivs = self.network.get_derivatives(grid_points)
            energy_density = self.energy_computer(derivs, grid_points)
        
        return {
            'coordinates': grid_points,
            'field_values': derivs['f'],
            'energy_density': energy_density,
            'field_gradient': derivs['f_grad'],
            'total_energy': energy_density.mean().item()
        }
    
    def plot_training_trace(self, save_path: Optional[str] = None):
        """Plot training convergence."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.training_history:
            warnings.warn("No training history to plot")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        total_loss = [h['total_loss'] for h in self.training_history]
        physics_loss = [h['physics_loss'] for h in self.training_history]
        constraint_loss = [h['constraint_loss'] for h in self.training_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, total_loss, 'k-', linewidth=2, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.semilogy(epochs, np.abs(total_loss), 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('|Loss| (log scale)')
        plt.title('Loss Magnitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, physics_loss, 'b-', label='Physics Loss')
        plt.plot(epochs, constraint_loss, 'g-', label='Constraint Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Components')
        plt.title('Loss Breakdown')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        improvement = np.array(total_loss) - total_loss[0]
        plt.plot(epochs, improvement, 'purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Improvement from Start')
        plt.title('Learning Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training trace saved to {save_path}")
        else:
            plt.show()


# Example energy computers
def mock_warp_bubble_energy_computer(derivs: Dict[str, torch.Tensor], 
                                   coords: torch.Tensor) -> torch.Tensor:
    """
    Mock energy computer for warp bubble configurations.
    
    This simulates the T₀₀ computation for a warp bubble metric.
    Replace with actual physics computation.
    """
    f = derivs['f']
    f_grad = derivs['f_grad']
    
    # Mock warp bubble energy density
    # This is a simplified model - real version would use full Einstein equations
    
    # Kinetic energy density (gradient terms)
    kinetic_density = torch.sum(f_grad**2, dim=1, keepdim=True)
    
    # Potential energy density (field magnitude)
    potential_density = f**2
    
    # Interaction terms
    r = torch.norm(coords, dim=1, keepdim=True)
    radial_factor = torch.exp(-r**2)
    
    # Total energy density (negative for exotic matter)
    T00 = -(kinetic_density + potential_density) * radial_factor
    
    return T00


def mock_quantum_field_energy_computer(derivs: Dict[str, torch.Tensor],
                                     coords: torch.Tensor) -> torch.Tensor:
    """
    Mock energy computer for quantum field configurations.
    """
    f = derivs['f']
    laplacian = derivs.get('laplacian', torch.zeros_like(f))
    
    # Quantum field energy density
    # Include kinetic term, potential, and nonlinear interactions
    
    kinetic_term = -0.5 * laplacian
    potential_term = 0.5 * f**2
    interaction_term = -0.1 * f**4  # Negative quartic (exotic matter)
    
    T00 = kinetic_term + potential_term + interaction_term
    
    return T00


# Example usage and testing
if __name__ == "__main__":
    print("=== Physics-Informed Neural Network for Exotic Matter ===")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
        exit(1)
    
    # Create network
    network = ProfileNet(
        input_dim=3,
        hidden_dim=32,  # Smaller for demo
        n_layers=3,
        output_dim=1,
        activation="tanh"
    )
    
    print(f"   • Network architecture: {network}")
    print(f"   • Parameters: {sum(p.numel() for p in network.parameters())}")
    
    # Create PINN optimizer
    pinn = ExoticMatterPINN(
        network=network,
        energy_computer=mock_warp_bubble_energy_computer,
        learning_rate=1e-3
    )
    
    print(f"   • Device: {pinn.device}")
    
    # Train the network
    result = pinn.train(
        n_epochs=500,
        batch_size=500,
        domain_bounds=(-1.0, 1.0),
        verbose=True
    )
    
    # Evaluate on test grid
    print(f"\n📊 EVALUATING OPTIMIZED FIELD:")
    
    # Create test grid
    n_test = 20
    x_test = torch.linspace(-1, 1, n_test)
    y_test = torch.linspace(-1, 1, n_test)
    z_test = torch.linspace(-1, 1, n_test)
    
    # Sample subset for evaluation (full grid would be large)
    test_coords = torch.rand(1000, 3) * 2 - 1  # Random points in [-1, 1]³
    test_coords = test_coords.to(pinn.device)
    
    field_evaluation = pinn.evaluate_field(test_coords)
    
    print(f"   • Total energy: {field_evaluation['total_energy']:.3e}")
    print(f"   • Field range: [{field_evaluation['field_values'].min():.3f}, {field_evaluation['field_values'].max():.3f}]")
    print(f"   • Energy density range: [{field_evaluation['energy_density'].min():.3e}, {field_evaluation['energy_density'].max():.3e}]")
    
    # Save results
    output_file = "pinn_exotic_matter_results.json"
    
    # Convert tensors to lists for JSON
    result_for_json = result.copy()
    result_for_json['test_evaluation'] = {
        'total_energy': field_evaluation['total_energy'],
        'field_min': field_evaluation['field_values'].min().item(),
        'field_max': field_evaluation['field_values'].max().item(),
        'energy_density_min': field_evaluation['energy_density'].min().item(),
        'energy_density_max': field_evaluation['energy_density'].max().item()
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_for_json, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot training if available
    try:
        pinn.plot_training_trace("pinn_training_trace.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Replace mock energy computer with real T₀₀ computation")
    print("2. Adjust network architecture for your problem")
    print("3. Add domain-specific physics constraints")
    print("4. Train with larger networks and longer times")
    print("5. Use optimized field in your exotic matter experiments")
