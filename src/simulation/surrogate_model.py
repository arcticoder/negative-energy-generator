"""
ML-Accelerated Surrogate Model for Fast Optimization
===================================================

This module implements machine learning surrogate models to accelerate
negative energy optimization across multiple physics domains.

The surrogate model learns from expensive simulations (FDTD, FEM, QFT)
to enable fast parameter space exploration and global optimization.

Mathematical Foundation:
    Gaussian Process: f(x) ~ GP(μ(x), k(x,x'))
    Acquisition function: α(x) = μ(x) - β√σ²(x)  (LCB)
    Expected improvement: EI(x) = σ(x)[ζΦ(ζ) + φ(ζ)]
    
    Neural network surrogate: f̂(x) = NN(x; θ)
    Uncertainty quantification via ensemble or MC-Dropout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable, Any
import warnings
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    acquisition_history: List[float]
    x_history: List[np.ndarray]
    f_history: List[float]

class PhysicsDataset(Dataset):
    """PyTorch dataset for physics simulation data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            X: Input parameters [n_samples × n_features]
            y: Target values [n_samples]
            transform: Optional data transformation
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.transform:
            x_sample = self.transform(x_sample)
        
        return x_sample, y_sample

class UncertaintyNet(nn.Module):
    """Neural network with uncertainty quantification via MC-Dropout."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int = 1, dropout_rate: float = 0.1):
        """
        Initialize uncertainty-aware neural network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1 for regression)
            dropout_rate: Dropout probability for uncertainty
        """
        super(UncertaintyNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        return self.network(x)
    
    def predict_with_uncertainty(self, x, n_samples: int = 100):
        """
        Predict with uncertainty via MC-Dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
        
        Returns:
            (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

class MultiPhysicsSurrogate:
    """
    Multi-physics surrogate model for negative energy optimization.
    
    Combines Gaussian Process and Neural Network surrogates for different
    physics domains (electromagnetic, quantum, mechanical).
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize multi-physics surrogate.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Surrogate models for different physics domains
        self.surrogates = {
            'electromagnetic': None,  # Will store GP or NN
            'quantum': None,
            'mechanical': None,
            'photonic': None
        }
        
        # Data scalers
        self.input_scalers = {}
        self.output_scalers = {}
        
        # Training history
        self.training_history = {}
        
        print(f"🧠 Multi-physics surrogate initialized")
        print(f"   • Device: {self.device}")
    
    def create_gaussian_process_surrogate(self, domain: str) -> GaussianProcessRegressor:
        """
        Create Gaussian Process surrogate for a physics domain.
        
        Args:
            domain: Physics domain name
        
        Returns:
            Configured GP surrogate
        """
        # Domain-specific kernel selection
        if domain == 'electromagnetic':
            # Smooth FDTD responses
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        elif domain == 'quantum':
            # Oscillatory quantum dynamics
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
        elif domain == 'mechanical':
            # Smooth mechanical responses
            kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-6)
        elif domain == 'photonic':
            # Band structure periodicity
            kernel = Matern(length_scale=0.8, nu=1.5) + WhiteKernel(noise_level=1e-5)
        else:
            # Default kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        return gp
    
    def create_neural_network_surrogate(self, input_dim: int, domain: str) -> UncertaintyNet:
        """
        Create neural network surrogate for a physics domain.
        
        Args:
            input_dim: Input parameter dimension
            domain: Physics domain name
        
        Returns:
            Configured NN surrogate
        """
        # Domain-specific architectures
        if domain == 'electromagnetic':
            # Deep network for complex field dynamics
            hidden_dims = [128, 64, 32, 16]
            dropout_rate = 0.1
        elif domain == 'quantum':
            # Medium network for quantum correlations
            hidden_dims = [64, 32, 16]
            dropout_rate = 0.15
        elif domain == 'mechanical':
            # Simple network for mechanical response
            hidden_dims = [32, 16, 8]
            dropout_rate = 0.05
        elif domain == 'photonic':
            # Medium network for band structure
            hidden_dims = [64, 32, 16]
            dropout_rate = 0.1
        else:
            # Default architecture
            hidden_dims = [64, 32, 16]
            dropout_rate = 0.1
        
        net = UncertaintyNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        return net
    
    def train_surrogate(self, domain: str, X_train: np.ndarray, y_train: np.ndarray,
                       model_type: str = 'gp', validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train surrogate model for a specific physics domain.
        
        Args:
            domain: Physics domain name
            X_train: Training inputs [n_samples × n_features]
            y_train: Training targets [n_samples]
            model_type: 'gp' for Gaussian Process, 'nn' for Neural Network
            validation_split: Fraction of data for validation
        
        Returns:
            Training metrics and model info
        """
        print(f"🏋️ Training {model_type.upper()} surrogate for {domain}")
        print(f"   • Training samples: {len(X_train)}")
        print(f"   • Input dimension: {X_train.shape[1]}")
        
        # Data preprocessing
        if domain not in self.input_scalers:
            self.input_scalers[domain] = StandardScaler()
            self.output_scalers[domain] = StandardScaler()
        
        X_scaled = self.input_scalers[domain].fit_transform(X_train)
        y_scaled = self.output_scalers[domain].fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train/validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        if model_type == 'gp':
            # Train Gaussian Process
            surrogate = self.create_gaussian_process_surrogate(domain)
            surrogate.fit(X_train_split, y_train_split)
            
            # Validation
            y_val_pred, y_val_std = surrogate.predict(X_val_split, return_std=True)
            val_mse = np.mean((y_val_split - y_val_pred)**2)
            val_mae = np.mean(np.abs(y_val_split - y_val_pred))
            val_nlpd = -np.mean(  # Negative log predictive density
                -0.5 * ((y_val_split - y_val_pred) / y_val_std)**2 - 
                0.5 * np.log(2 * np.pi * y_val_std**2)
            )
            
            metrics = {
                'validation_mse': val_mse,
                'validation_mae': val_mae,
                'validation_nlpd': val_nlpd,
                'model_type': 'gaussian_process'
            }
            
        elif model_type == 'nn':
            # Train Neural Network
            surrogate = self.create_neural_network_surrogate(X_train.shape[1], domain)
            
            # Create data loaders
            train_dataset = PhysicsDataset(X_train_split, y_train_split)
            val_dataset = PhysicsDataset(X_val_split, y_val_split)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(surrogate.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            n_epochs = 200
            train_losses = []
            val_losses = []
            
            for epoch in range(n_epochs):
                # Training
                surrogate.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = surrogate(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                surrogate.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = surrogate(batch_x).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 50 == 0:
                    print(f"   • Epoch {epoch+1}: Train loss = {train_loss:.6f}, Val loss = {val_loss:.6f}")
            
            metrics = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'train_history': train_losses,
                'val_history': val_losses,
                'model_type': 'neural_network'
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store surrogate
        self.surrogates[domain] = surrogate
        self.training_history[domain] = metrics
        
        print(f"   ✅ Training complete")
        if model_type == 'gp':
            print(f"   • Validation MSE: {metrics['validation_mse']:.6f}")
        else:
            print(f"   • Final validation loss: {metrics['final_val_loss']:.6f}")
        
        return metrics
    
    def predict(self, domain: str, X: np.ndarray, 
                return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using trained surrogate.
        
        Args:
            domain: Physics domain
            X: Input parameters [n_samples × n_features]
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            (predictions, uncertainties) or just predictions
        """
        if domain not in self.surrogates or self.surrogates[domain] is None:
            raise ValueError(f"No trained surrogate for domain: {domain}")
        
        surrogate = self.surrogates[domain]
        
        # Scale inputs
        X_scaled = self.input_scalers[domain].transform(X)
        
        if isinstance(surrogate, GaussianProcessRegressor):
            # GP prediction
            if return_uncertainty:
                y_pred_scaled, y_std_scaled = surrogate.predict(X_scaled, return_std=True)
                
                # Unscale outputs
                y_pred = self.output_scalers[domain].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_std = y_std_scaled * self.output_scalers[domain].scale_[0]
                
                return y_pred, y_std
            else:
                y_pred_scaled = surrogate.predict(X_scaled)
                y_pred = self.output_scalers[domain].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                return y_pred, None
        
        else:  # Neural Network
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            if return_uncertainty:
                y_pred_scaled, y_std_scaled = surrogate.predict_with_uncertainty(X_tensor)
                
                # Convert to numpy and unscale
                y_pred_scaled = y_pred_scaled.cpu().numpy().ravel()
                y_std_scaled = y_std_scaled.cpu().numpy().ravel()
                
                y_pred = self.output_scalers[domain].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_std = y_std_scaled * self.output_scalers[domain].scale_[0]
                
                return y_pred, y_std
            else:
                surrogate.eval()
                with torch.no_grad():
                    y_pred_scaled = surrogate(X_tensor).cpu().numpy().ravel()
                
                y_pred = self.output_scalers[domain].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                return y_pred, None

def bayesian_optimization(objective_function: Callable,
                         bounds: List[Tuple[float, float]],
                         n_initial: int = 10,
                         n_iterations: int = 50,
                         acquisition: str = 'lcb',
                         beta: float = 2.0) -> OptimizationResult:
    """
    Bayesian optimization using Gaussian Process surrogate.
    
    Args:
        objective_function: Function to minimize
        bounds: Parameter bounds [(low, high), ...]
        n_initial: Number of initial random samples
        n_iterations: Number of BO iterations
        acquisition: Acquisition function ('lcb', 'ei', 'pi')
        beta: Exploration parameter for LCB
    
    Returns:
        Optimization result
    """
    print(f"🎯 BAYESIAN OPTIMIZATION")
    print(f"   • Bounds: {bounds}")
    print(f"   • Initial samples: {n_initial}")
    print(f"   • Iterations: {n_iterations}")
    print(f"   • Acquisition: {acquisition}")
    
    # Initial random sampling
    X_samples = []
    y_samples = []
    
    for _ in range(n_initial):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        y = objective_function(x)
        X_samples.append(x)
        y_samples.append(y)
    
    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)
    
    # Initialize GP surrogate
    surrogate = MultiPhysicsSurrogate(use_gpu=False)  # Use CPU for small GP
    
    acquisition_history = []
    x_history = list(X_samples)
    f_history = list(y_samples)
    
    for iteration in range(n_iterations):
        print(f"   • Iteration {iteration+1}/{n_iterations}")
        
        # Train GP surrogate
        surrogate.train_surrogate('optimization', X_samples, y_samples, model_type='gp')
        
        # Optimize acquisition function
        best_acq = -np.inf
        best_x = None
        
        # Grid search for acquisition optimization (simple approach)
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            x_candidate = np.array([np.random.uniform(low, high) for low, high in bounds])
            candidates.append(x_candidate)
        
        candidates = np.array(candidates)
        
        # Evaluate acquisition function
        y_pred, y_std = surrogate.predict('optimization', candidates, return_uncertainty=True)
        
        if acquisition == 'lcb':
            # Lower Confidence Bound
            acq_values = y_pred - beta * y_std
        elif acquisition == 'ei':
            # Expected Improvement
            f_best = np.min(y_samples)
            z = (f_best - y_pred) / (y_std + 1e-9)
            acq_values = (f_best - y_pred) * norm.cdf(z) + y_std * norm.pdf(z)
        else:  # Default to LCB
            acq_values = y_pred - beta * y_std
        
        # Select best candidate
        best_idx = np.argmax(acq_values) if acquisition == 'ei' else np.argmin(acq_values)
        x_next = candidates[best_idx]
        
        # Evaluate objective
        y_next = objective_function(x_next)
        
        # Update data
        X_samples = np.vstack([X_samples, x_next])
        y_samples = np.append(y_samples, y_next)
        
        acquisition_history.append(acq_values[best_idx])
        x_history.append(x_next)
        f_history.append(y_next)
        
        print(f"     → x_next: {x_next}")
        print(f"     → f_next: {y_next:.6f}")
    
    # Find best result
    best_idx = np.argmin(y_samples)
    x_opt = X_samples[best_idx]
    f_opt = y_samples[best_idx]
    
    print(f"   ✅ Optimization complete")
    print(f"   • Optimal x: {x_opt}")
    print(f"   • Optimal f: {f_opt:.6f}")
    
    return OptimizationResult(
        x_opt=x_opt,
        f_opt=f_opt,
        iterations=n_iterations,
        acquisition_history=acquisition_history,
        x_history=x_history,
        f_history=f_history
    )

def multi_domain_optimization(objective_functions: Dict[str, Callable],
                             bounds: List[Tuple[float, float]],
                             weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Multi-domain optimization combining multiple physics objectives.
    
    Args:
        objective_functions: Dict of domain -> objective function
        bounds: Parameter bounds
        weights: Relative weights for each domain
    
    Returns:
        Multi-domain optimization results
    """
    print("🌐 MULTI-DOMAIN OPTIMIZATION")
    print("=" * 60)
    
    domains = list(objective_functions.keys())
    if weights is None:
        weights = {domain: 1.0 for domain in domains}
    
    print(f"   • Domains: {domains}")
    print(f"   • Weights: {weights}")
    
    # Combined objective function
    def combined_objective(x):
        total = 0.0
        for domain, obj_func in objective_functions.items():
            value = obj_func(x)
            total += weights[domain] * value
        return total
    
    # Run Bayesian optimization
    result = bayesian_optimization(
        objective_function=combined_objective,
        bounds=bounds,
        n_initial=15,
        n_iterations=75
    )
    
    # Evaluate individual domains at optimum
    domain_results = {}
    for domain, obj_func in objective_functions.items():
        domain_value = obj_func(result.x_opt)
        domain_results[domain] = domain_value
    
    print(f"\n✅ MULTI-DOMAIN OPTIMIZATION COMPLETE")
    print(f"   • Combined optimum: {result.f_opt:.6f}")
    for domain, value in domain_results.items():
        print(f"   • {domain}: {value:.6f}")
    
    return {
        'optimization_result': result,
        'domain_results': domain_results,
        'weights': weights,
        'combined_optimum': result.f_opt
    }

# Demo functions
def mock_electromagnetic_objective(x):
    """Mock electromagnetic FDTD objective (Casimir energy)."""
    return -np.exp(-np.sum((x - 0.3)**2)) + 0.1 * np.sum(x**2)

def mock_quantum_objective(x):
    """Mock quantum circuit objective (negative energy extraction)."""
    return np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 0.05 * np.sum(x**2)

def mock_mechanical_objective(x):
    """Mock mechanical FEM objective (plate stability)."""
    return (x[0] - 0.5)**2 + (x[1] - 0.8)**2 - 0.1

def generate_training_data(objective_func: Callable, bounds: List[Tuple[float, float]], 
                          n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for surrogate model."""
    X = []
    y = []
    
    for _ in range(n_samples):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        y_val = objective_func(x)
        X.append(x)
        y.append(y_val)
    
    return np.array(X), np.array(y)

def run_surrogate_demo():
    """Run comprehensive surrogate model demonstration."""
    print("🚀 SURROGATE MODEL DEMO")
    print("=" * 50)
    
    # Problem bounds
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Generate training data for each domain
    X_em, y_em = generate_training_data(mock_electromagnetic_objective, bounds, 150)
    X_qm, y_qm = generate_training_data(mock_quantum_objective, bounds, 150)
    X_mech, y_mech = generate_training_data(mock_mechanical_objective, bounds, 150)
    
    # Create and train surrogates
    surrogate = MultiPhysicsSurrogate()
    
    # Train GP surrogates
    surrogate.train_surrogate('electromagnetic', X_em, y_em, model_type='gp')
    surrogate.train_surrogate('quantum', X_qm, y_qm, model_type='nn')
    surrogate.train_surrogate('mechanical', X_mech, y_mech, model_type='gp')
    
    # Test predictions
    X_test = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    
    for domain in ['electromagnetic', 'quantum', 'mechanical']:
        y_pred, y_std = surrogate.predict(domain, X_test, return_uncertainty=True)
        print(f"\n{domain.upper()} predictions:")
        for i, (x_test, pred, std) in enumerate(zip(X_test, y_pred, y_std)):
            print(f"   • x={x_test} → f={pred:.3f} ± {std:.3f}")
    
    # Run multi-domain optimization
    objective_functions = {
        'electromagnetic': mock_electromagnetic_objective,
        'quantum': mock_quantum_objective,
        'mechanical': mock_mechanical_objective
    }
    
    multi_result = multi_domain_optimization(objective_functions, bounds)
    
    return surrogate, multi_result

if __name__ == "__main__":
    # Import additional dependencies
    try:
        from scipy.stats import norm
    except ImportError:
        print("⚠️  scipy not available, using approximation")
        class norm:
            @staticmethod
            def cdf(x):
                return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
            @staticmethod
            def pdf(x):
                return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    # Run demonstration
    surrogate_model, optimization_results = run_surrogate_demo()
    
    print(f"\n🎯 Final multi-domain optimum: {optimization_results['combined_optimum']:.6f}")
    print("✅ Surrogate model demo complete!")
