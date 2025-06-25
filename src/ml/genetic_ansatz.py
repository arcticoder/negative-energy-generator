"""
Genetic (Evolutionary) Search of Spin-Network Couplings
======================================================

This module implements evolutionary optimization of coupling matrices K_ij
and ansatz coefficients using genetic algorithms. Treats parameters as "genes"
and evolves populations toward maximal negative energy.

Mathematical Foundation:
    Population {K^(p)}, fitness F(K) = ∫ T₀₀[G(K)] d³x
    Uses selection, crossover, and mutation to explore 10³-10⁵ dimensional space
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Callable, Optional, Any
import json
from datetime import datetime
from pathlib import Path

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    warnings.warn("DEAP not available. Install with: pip install deap")
    DEAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class GeneticAnsatzOptimizer:
    """
    Genetic algorithm optimizer for exotic matter coupling matrices and ansätze.
    
    Evolves populations of parameter vectors toward configurations that
    maximize negative energy density.
    """
    
    def __init__(self, energy_evaluator: Callable, genome_length: int,
                 population_size: int = 50, gene_bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize the genetic optimizer.
        
        Args:
            energy_evaluator: Function that takes parameter vector and returns energy
            genome_length: Length of parameter vector (number of genes)
            population_size: Size of evolution population
            gene_bounds: (min, max) bounds for gene values
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP required for genetic algorithms")
        
        self.energy_evaluator = energy_evaluator
        self.genome_length = genome_length
        self.population_size = population_size
        self.gene_bounds = gene_bounds
        self.evolution_history = []
        self.best_individual = None
        self.best_fitness = -np.inf  # We want maximum negative energy
        
        # Setup DEAP framework
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        # Clear any existing classes
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create fitness and individual classes
        # We want to MAXIMIZE negative energy (minimize positive values)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Gene generation
        min_val, max_val = self.gene_bounds
        self.toolbox.register("attr_float", np.random.uniform, min_val, max_val)
        
        # Individual and population creation
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, n=self.genome_length)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, 
                             mu=0, sigma=(max_val-min_val)*0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: List of gene values (parameters)
            
        Returns:
            Tuple containing fitness value
        """
        try:
            # Evaluate energy for this parameter configuration
            energy = self.energy_evaluator(individual)
            
            # For genetic algorithms, we want to MAXIMIZE the fitness
            # Since we want negative energy, we return the energy directly
            # (more negative = higher fitness for maximization)
            fitness = energy
            
            # Track best individual
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()
            
            return (fitness,)
            
        except Exception as e:
            warnings.warn(f"Evaluation failed for individual {individual[:3]}...: {e}")
            return (-1e6,)  # Large negative penalty
    
    def evolve(self, n_generations: int = 50, crossover_prob: float = 0.5,
               mutation_prob: float = 0.2, verbose: bool = True) -> Dict:
        """
        Run genetic algorithm evolution.
        
        Args:
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evolution results
        """
        if verbose:
            print(f"🧬 Starting Genetic Evolution for Exotic Matter Optimization")
            print(f"   • Population size: {self.population_size}")
            print(f"   • Genome length: {self.genome_length}")
            print(f"   • Generations: {n_generations}")
            print(f"   • Crossover prob: {crossover_prob}")
            print(f"   • Mutation prob: {mutation_prob}")
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Track evolution
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Initial statistics
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        
        if verbose:
            print(f"   Gen 0: avg={record['avg']:.3e}, best={record['max']:.3e}")
        
        # Main evolution loop
        for generation in range(1, n_generations + 1):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if np.random.random() < mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=generation, nevals=len(invalid_ind), **record)
            
            # Store generation in history
            generation_data = {
                'generation': generation,
                'avg_fitness': record['avg'],
                'best_fitness': record['max'],
                'std_fitness': record['std'],
                'best_individual': tools.selBest(population, 1)[0][:].copy()
            }
            self.evolution_history.append(generation_data)
            
            if verbose and generation % 10 == 0:
                print(f"   Gen {generation}: avg={record['avg']:.3e}, best={record['max']:.3e}")
        
        # Final results
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        
        evolution_result = {
            'success': True,
            'best_individual': best_individual[:].copy(),
            'best_fitness': best_fitness,
            'final_population': [ind[:].copy() for ind in population],
            'evolution_history': self.evolution_history,
            'logbook': {
                'generations': logbook.select("gen"),
                'avg_fitness': logbook.select("avg"),
                'max_fitness': logbook.select("max"),
                'min_fitness': logbook.select("min"),
                'std_fitness': logbook.select("std")
            },
            'parameters': {
                'population_size': self.population_size,
                'genome_length': self.genome_length,
                'n_generations': n_generations,
                'crossover_prob': crossover_prob,
                'mutation_prob': mutation_prob,
                'gene_bounds': self.gene_bounds
            }
        }
        
        if verbose:
            print(f"✅ Evolution completed!")
            print(f"   • Best fitness (energy): {best_fitness:.3e} J")
            print(f"   • Best individual: {best_individual[:5]}... (showing first 5 genes)")
        
        return evolution_result
    
    def analyze_genetic_diversity(self, population: Optional[List] = None) -> Dict:
        """
        Analyze genetic diversity in the population.
        
        Args:
            population: Population to analyze (uses final if None)
            
        Returns:
            Dictionary with diversity metrics
        """
        if population is None and not self.evolution_history:
            warnings.warn("No evolution history available for diversity analysis")
            return {}
        
        if population is None:
            # Use final population from last generation
            population = self.evolution_history[-1]['best_individual']
            if isinstance(population, list) and len(population) > 0:
                # This is actually a single individual, not population
                return {'note': 'Single individual analysis not implemented'}
        
        # Convert to numpy array for analysis
        pop_array = np.array([ind if isinstance(ind, list) else ind[:] for ind in population])
        
        # Calculate diversity metrics
        diversity_metrics = {
            'mean_genes': np.mean(pop_array, axis=0).tolist(),
            'std_genes': np.std(pop_array, axis=0).tolist(),
            'gene_ranges': (np.max(pop_array, axis=0) - np.min(pop_array, axis=0)).tolist(),
            'population_variance': np.var(pop_array),
            'total_diversity': np.sum(np.var(pop_array, axis=0))
        }
        
        return diversity_metrics
    
    def plot_evolution_trace(self, save_path: Optional[str] = None):
        """Plot the evolution convergence trace."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("matplotlib not available for plotting")
            return
        
        if not self.evolution_history:
            warnings.warn("No evolution history to plot")
            return
        
        generations = [h['generation'] for h in self.evolution_history]
        avg_fitness = [h['avg_fitness'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        std_fitness = [h['std_fitness'] for h in self.evolution_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        plt.plot(generations, best_fitness, 'r-', label='Best Fitness')
        plt.fill_between(generations, 
                        np.array(avg_fitness) - np.array(std_fitness),
                        np.array(avg_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue', label='±1σ')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Energy J)')
        plt.title('Evolution Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.semilogy(-np.array(best_fitness), 'g-', linewidth=2, label='|Best Energy|')
        plt.xlabel('Generation')
        plt.ylabel('|Energy| (J, log scale)')
        plt.title('Best Energy Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(generations, std_fitness, 'm-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Population Std Dev')
        plt.title('Genetic Diversity')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        improvement = np.array(best_fitness) - best_fitness[0]
        plt.plot(generations, improvement, 'orange', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Improvement from Gen 0')
        plt.title('Fitness Improvement')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evolution trace saved to {save_path}")
        else:
            plt.show()


def create_coupling_matrix_genome(matrix_size: int, symmetric: bool = True) -> int:
    """
    Calculate genome length for coupling matrix optimization.
    
    Args:
        matrix_size: Size of coupling matrix (N×N)
        symmetric: Whether matrix should be symmetric
        
    Returns:
        Required genome length
    """
    if symmetric:
        # Only need upper triangle + diagonal
        return matrix_size * (matrix_size + 1) // 2
    else:
        # Need full matrix
        return matrix_size * matrix_size


def genome_to_coupling_matrix(genome: List[float], matrix_size: int, 
                            symmetric: bool = True) -> np.ndarray:
    """
    Convert genome to coupling matrix.
    
    Args:
        genome: List of gene values
        matrix_size: Size of output matrix
        symmetric: Whether to make matrix symmetric
        
    Returns:
        Coupling matrix
    """
    if symmetric:
        # Fill upper triangle + diagonal
        matrix = np.zeros((matrix_size, matrix_size))
        idx = 0
        for i in range(matrix_size):
            for j in range(i, matrix_size):
                matrix[i, j] = genome[idx]
                if i != j:
                    matrix[j, i] = genome[idx]  # Make symmetric
                idx += 1
        return matrix
    else:
        # Fill full matrix
        return np.array(genome).reshape((matrix_size, matrix_size))


# Example usage and testing
if __name__ == "__main__":
    print("=== Genetic Evolution of Exotic Matter Couplings ===")
    
    if not DEAP_AVAILABLE:
        print("❌ DEAP not available. Install with: pip install deap")
        exit(1)
    
    # Mock energy evaluator for testing
    def mock_coupling_energy_evaluator(genome):
        """
        Mock function for coupling matrix energy evaluation.
        
        This would be replaced with actual T₀₀ computation using the
        coupling matrix derived from the genome.
        """
        # Convert genome to 8×8 coupling matrix
        matrix_size = 8
        K = genome_to_coupling_matrix(genome, matrix_size, symmetric=True)
        
        # Simulate exotic matter energy calculation
        # This is a toy model - real version would use your T₀₀ computation
        
        # Penalty for too large eigenvalues (stability)
        eigenvals = np.linalg.eigvals(K)
        stability_penalty = np.sum(np.maximum(0, np.real(eigenvals) - 0.8)**2)
        
        # Reward for specific coupling patterns that might generate negative energy
        # Target: sparse matrix with specific structure
        sparsity_bonus = -np.sum(np.abs(K)) * 0.1
        structure_bonus = -np.trace(K**2) * 0.05
        
        # Random component to simulate complex physics
        physics_term = -np.sum(np.sin(K.flatten() * 3)) * 0.02
        
        total_energy = sparsity_bonus + structure_bonus + physics_term - stability_penalty
        
        # Add some noise
        total_energy += np.random.normal(0, 0.001)
        
        return total_energy
    
    # Setup genetic optimization for 8×8 symmetric coupling matrix
    matrix_size = 8
    genome_length = create_coupling_matrix_genome(matrix_size, symmetric=True)
    
    print(f"   • Matrix size: {matrix_size}×{matrix_size}")
    print(f"   • Genome length: {genome_length}")
    print(f"   • Symmetric matrix: Yes")
    
    # Initialize genetic optimizer
    optimizer = GeneticAnsatzOptimizer(
        energy_evaluator=mock_coupling_energy_evaluator,
        genome_length=genome_length,
        population_size=60,
        gene_bounds=(-0.5, 0.5)
    )
    
    # Run evolution
    result = optimizer.evolve(
        n_generations=40,
        crossover_prob=0.6,
        mutation_prob=0.15,
        verbose=True
    )
    
    # Analyze best individual
    best_genome = result['best_individual']
    best_matrix = genome_to_coupling_matrix(best_genome, matrix_size, symmetric=True)
    
    print(f"\n📊 EVOLUTION RESULTS:")
    print(f"   • Best fitness (energy): {result['best_fitness']:.3e} J")
    print(f"   • Best coupling matrix shape: {best_matrix.shape}")
    print(f"   • Matrix eigenvalues: {np.linalg.eigvals(best_matrix)[:3].real}... (first 3)")
    print(f"   • Matrix trace: {np.trace(best_matrix):.3f}")
    print(f"   • Matrix norm: {np.linalg.norm(best_matrix):.3f}")
    
    # Diversity analysis
    diversity = optimizer.analyze_genetic_diversity()
    if diversity and 'total_diversity' in diversity:
        print(f"   • Final population diversity: {diversity['total_diversity']:.3f}")
    
    # Save results
    output_file = "genetic_coupling_optimization_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    result_for_json = result.copy()
    result_for_json['best_coupling_matrix'] = best_matrix.tolist()
    result_for_json['best_eigenvalues'] = np.linalg.eigvals(best_matrix).real.tolist()
    
    with open(output_file, 'w') as f:
        json.dump(result_for_json, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Plot evolution if available
    try:
        optimizer.plot_evolution_trace("genetic_evolution_trace.png")
    except:
        print("   (Plotting not available)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Replace mock evaluator with real T₀₀ computation")
    print("2. Adjust matrix size and genome bounds for your system")
    print("3. Run with larger populations and more generations")
    print("4. Use best coupling matrix in your exotic matter simulator")
    print("5. Experiment with different genetic operators")
