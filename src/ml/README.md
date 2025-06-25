# Machine Learning Optimization Modules
================================

This directory contains advanced machine learning-based optimization modules for exotic matter research and negative energy systems. These modules complement the core prototype stack with intelligent optimization algorithms.

## 📁 Module Overview

### `bo_ansatz_opt.py` - Bayesian Optimization
**Gaussian Process-based optimization for ansatz parameters**

- **Purpose**: Optimize coupling matrices, field configurations, and experimental parameters using Bayesian optimization
- **Key Features**:
  - Scikit-optimize (skopt) integration
  - Acquisition function optimization (Expected Improvement, UCB)
  - Multi-dimensional parameter spaces
  - Uncertainty quantification
  - Convergence analysis and visualization

- **Usage**:
  ```python
  from src.ml.bo_ansatz_opt import BayesianAnsatzOptimizer
  
  # Define parameter bounds
  bounds = [(-1.0, 1.0), (-2.0, 2.0), (0.1, 10.0)]
  
  # Initialize optimizer
  optimizer = BayesianAnsatzOptimizer(bounds)
  
  # Run optimization
  result = optimizer.optimize(
      objective_function=your_T00_evaluator,
      n_calls=100,
      random_state=42
  )
  ```

### `genetic_ansatz.py` - Genetic Algorithm Optimization
**Evolutionary optimization for complex parameter landscapes**

- **Purpose**: Optimize ansatz coefficients and coupling matrices using genetic algorithms
- **Key Features**:
  - DEAP framework integration
  - Multi-objective optimization support
  - Population diversity management
  - Crossover and mutation operators
  - Convergence tracking and analysis

- **Usage**:
  ```python
  from src.ml.genetic_ansatz import GeneticAnsatzOptimizer
  
  # Initialize optimizer
  optimizer = GeneticAnsatzOptimizer(
      n_parameters=10,
      bounds=[(-5.0, 5.0)] * 10
  )
  
  # Run evolution
  result = optimizer.evolve(
      fitness_function=your_fitness_evaluator,
      n_generations=50,
      population_size=100
  )
  ```

### `pinn_exotic.py` - Physics-Informed Neural Networks
**Deep learning with physics constraints for direct T₀₀ optimization**

- **Purpose**: Train neural networks to parameterize field configurations that maximize negative energy
- **Key Features**:
  - PyTorch-based implementation
  - Physics-informed loss functions
  - Automatic differentiation for constraint enforcement
  - Real-time field evaluation
  - Training visualization and analysis

- **Usage**:
  ```python
  from src.ml.pinn_exotic import ProfileNet, ExoticMatterPINN
  
  # Create network
  network = ProfileNet(input_dim=3, hidden_dim=64, n_layers=4)
  
  # Initialize PINN
  pinn = ExoticMatterPINN(
      network=network,
      energy_computer=your_T00_computer
  )
  
  # Train
  result = pinn.train(n_epochs=1000, batch_size=1000)
  ```

## 🔧 Dependencies

The ML modules require additional dependencies beyond the core prototype stack:

```bash
pip install scikit-optimize deap torch matplotlib seaborn
```

**Core Dependencies:**
- `scikit-optimize` (skopt): Bayesian optimization
- `deap`: Genetic algorithms and evolutionary computation
- `torch`: Deep learning and automatic differentiation
- `matplotlib`: Plotting and visualization
- `seaborn`: Advanced statistical plots

**Optional Dependencies:**
- `tensorboard`: Training visualization for PINN
- `plotly`: Interactive plots
- `hyperopt`: Alternative Bayesian optimization backend

## 🚀 Quick Start

### 1. Basic Bayesian Optimization
```python
import numpy as np
from src.ml.bo_ansatz_opt import BayesianAnsatzOptimizer

# Define a mock T₀₀ evaluator
def mock_energy_evaluator(params):
    # Simulate T₀₀ calculation
    x, y, z = params
    return -(x**2 + y**2 + z**2)  # Negative for energy minimization

# Optimize
optimizer = BayesianAnsatzOptimizer([(-2, 2), (-2, 2), (-2, 2)])
result = optimizer.optimize(mock_energy_evaluator, n_calls=50)
print(f"Best parameters: {result.x}")
print(f"Best energy: {result.fun}")
```

### 2. Genetic Algorithm Optimization
```python
from src.ml.genetic_ansatz import GeneticAnsatzOptimizer

def fitness_function(individual):
    # Convert to T₀₀ (higher fitness = more negative energy)
    energy = sum(x**2 for x in individual)
    return (-energy,)  # Return tuple for DEAP

optimizer = GeneticAnsatzOptimizer(n_parameters=5, bounds=[(-3, 3)] * 5)
result = optimizer.evolve(fitness_function, n_generations=30)
```

### 3. Physics-Informed Neural Network
```python
from src.ml.pinn_exotic import ProfileNet, ExoticMatterPINN, mock_warp_bubble_energy_computer

# Create and train PINN
network = ProfileNet(input_dim=3, hidden_dim=32, n_layers=3)
pinn = ExoticMatterPINN(network, mock_warp_bubble_energy_computer)
result = pinn.train(n_epochs=500, batch_size=500)

# Evaluate optimized field
import torch
test_points = torch.rand(100, 3) * 2 - 1  # Random points in [-1,1]³
field_evaluation = pinn.evaluate_field(test_points)
print(f"Optimized energy: {field_evaluation['total_energy']}")
```

## 🎯 Integration with Prototype Stack

The ML modules integrate seamlessly with the core prototype stack:

### With Exotic Matter Simulator
```python
from src.prototype.exotic_matter_simulator import ExoticMatterSimulator
from src.ml.bo_ansatz_opt import BayesianAnsatzOptimizer

# Create simulator
simulator = ExoticMatterSimulator(grid_size=50, domain_size=2.0)

# Define optimization objective
def optimize_field_config(params):
    alpha, beta, gamma = params
    # Set field configuration
    simulator.set_field_ansatz(alpha, beta, gamma)
    # Compute T₀₀
    T00_field = simulator.compute_stress_energy_tensor()
    return simulator.compute_total_energy(T00_field)

# Optimize
optimizer = BayesianAnsatzOptimizer([(-1, 1), (-1, 1), (-1, 1)])
result = optimizer.optimize(optimize_field_config, n_calls=100)
```

### With Measurement Pipeline
```python
from src.prototype.measurement_pipeline import MeasurementPipeline
from src.ml.genetic_ansatz import GeneticAnsatzOptimizer

# Create measurement pipeline
pipeline = MeasurementPipeline()

# Optimize experimental parameters
def fitness_from_measurements(params):
    # Configure experiment
    pipeline.configure_experiment(params)
    # Run measurement
    data = pipeline.generate_mock_data(1000)
    # Analyze results
    analysis = pipeline.analyze_energy_data(data)
    return (analysis['negative_energy_fraction'],)

optimizer = GeneticAnsatzOptimizer(n_parameters=4, bounds=[(0, 1)] * 4)
result = optimizer.evolve(fitness_from_measurements, n_generations=25)
```

## 📊 Performance Comparison

The ML modules are designed to outperform traditional optimization methods:

| Method | Convergence Speed | Global Optima | Parameter Scaling | Physics Integration |
|--------|------------------|---------------|-------------------|---------------------|
| **Bayesian Opt** | Fast | Good | Excellent | Moderate |
| **Genetic Algorithm** | Moderate | Excellent | Good | Moderate |
| **PINN** | Slow | Good | Excellent | Excellent |
| Grid Search | Very Slow | Poor | Poor | None |
| Random Search | Slow | Poor | Moderate | None |

## 🔬 Advanced Features

### Multi-Objective Optimization
```python
# Optimize both energy and stability simultaneously
def multi_objective_fitness(individual):
    energy = compute_T00(individual)
    stability = compute_field_stability(individual)
    return (energy, stability)  # Pareto optimization

optimizer = GeneticAnsatzOptimizer(n_parameters=6, bounds=[(-2, 2)] * 6)
optimizer.configure_multi_objective(['minimize', 'maximize'])
result = optimizer.evolve(multi_objective_fitness, n_generations=40)
```

### Constraint Handling
```python
# Add physics constraints to PINN
def constrained_energy_computer(derivs, coords):
    # Standard energy computation
    T00 = standard_energy_computer(derivs, coords)
    
    # Add constraint penalties
    field = derivs['f']
    constraint_penalty = torch.mean(torch.relu(field - 1.0)**2)  # Field ≤ 1
    
    return T00 + 0.1 * constraint_penalty
```

### Transfer Learning
```python
# Use pre-trained network for new problems
pretrained_network = ProfileNet.load_state_dict(torch.load('best_model.pth'))
new_pinn = ExoticMatterPINN(pretrained_network, new_energy_computer)
# Fine-tune with fewer epochs
result = new_pinn.train(n_epochs=100, learning_rate=1e-4)
```

## 🎛️ Hyperparameter Tuning

### Bayesian Optimization Settings
```python
# Acquisition function comparison
acquisition_functions = ['EI', 'PI', 'UCB']
for acq_func in acquisition_functions:
    optimizer = BayesianAnsatzOptimizer(bounds, acquisition_function=acq_func)
    result = optimizer.optimize(objective, n_calls=50)
    print(f"{acq_func}: {result.fun}")
```

### Genetic Algorithm Tuning
```python
# Population size vs. convergence
for pop_size in [50, 100, 200]:
    optimizer = GeneticAnsatzOptimizer(n_parameters=10, bounds=bounds)
    result = optimizer.evolve(fitness_func, n_generations=30, population_size=pop_size)
    print(f"Pop {pop_size}: Best fitness = {result['best_fitness']}")
```

### PINN Architecture Search
```python
# Compare network architectures
architectures = [
    {'hidden_dim': 32, 'n_layers': 3},
    {'hidden_dim': 64, 'n_layers': 4},
    {'hidden_dim': 128, 'n_layers': 5}
]

for arch in architectures:
    network = ProfileNet(**arch)
    pinn = ExoticMatterPINN(network, energy_computer)
    result = pinn.train(n_epochs=500)
    print(f"Arch {arch}: Final loss = {result['best_loss']}")
```

## 🚀 Next Steps

1. **Ensemble Methods**: Combine multiple ML approaches for robust optimization
2. **Active Learning**: Intelligently select measurement points for maximum information gain
3. **Reinforcement Learning**: Train agents for real-time experimental control
4. **Meta-Learning**: Learn to optimize optimization algorithms themselves
5. **Quantum ML**: Integrate quantum computing for enhanced optimization

## 📚 References

- **Bayesian Optimization**: Mockus, J. (1989). Bayesian Approach to Global Optimization
- **Genetic Algorithms**: Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
- **Physics-Informed Neural Networks**: Raissi, M. et al. (2019). Physics-informed neural networks
- **Exotic Matter Theory**: Morris, M. S. & Thorne, K. S. (1988). Wormholes in spacetime

---

*The ML optimization modules provide state-of-the-art intelligent optimization for pushing the boundaries of exotic matter research and negative energy generation.*
