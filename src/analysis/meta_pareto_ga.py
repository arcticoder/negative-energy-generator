# File: src/analysis/meta_pareto_ga.py
"""
Multi-Objective Genetic Algorithm for Metamaterial Stacking Optimization

Mathematical Foundation:
E_meta = E₀ · √N · 1/(1 + α·δa/a + β·δf)
Cost(N) = N

NSGA-II optimization to maximize energy and minimize layer count.
"""

import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("⚠️  DEAP not available - using simplified GA fallback")

try:
    from src.analysis.in_silico_stack_and_squeeze import simulate_photonic_metamaterial
    INSILICO_AVAILABLE = True
except ImportError:
    INSILICO_AVAILABLE = False
    print("⚠️  in_silico_stack_and_squeeze not available - using fallback")

def simulate_photonic_metamaterial_fallback(lattice_const: float, filling_fraction: float, n_layers: int):
    """Fallback metamaterial simulation if main module unavailable."""
    # Simple physics model
    optimal_lattice = 250e-9
    optimal_filling = 0.35
    base_energy = -1e-15
    
    # Geometric detunings
    lattice_detuning = abs(lattice_const - optimal_lattice) / optimal_lattice
    filling_detuning = abs(filling_fraction - optimal_filling) / optimal_filling
    
    # Enhancement with fabrication penalties
    alpha, beta = 2.0, 5.0
    geometric_factor = 1 / (1 + alpha * lattice_detuning + beta * filling_detuning)
    
    # Layer enhancement with saturation
    if n_layers >= 10:
        layer_factor = np.sum([0.95 * k**(-0.5) for k in range(1, n_layers + 1)])
    else:
        layer_factor = np.sqrt(n_layers)
    
    total_energy = base_energy * geometric_factor * layer_factor
    
    return {
        'total_negative_energy': total_energy,
        'enhancement_factor': geometric_factor * layer_factor,
        'layer_factor': layer_factor,
        'fabrication_score': geometric_factor
    }

# Use available metamaterial function
if INSILICO_AVAILABLE:
    metamaterial_func = simulate_photonic_metamaterial
else:
    metamaterial_func = simulate_photonic_metamaterial_fallback

if DEAP_AVAILABLE:
    # Real DEAP implementation
    print("✅ Using DEAP for NSGA-II multi-objective optimization")
    
    # 1) Create fitness: maximize energy, minimize layers
    creator.create("FitnessMeta", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMeta)
    
    toolbox = base.Toolbox()
    
    # Decision vars: lattice (100–500 nm), fill_frac (0.1–0.5), layers (1–20)
    toolbox.register("attr_lattice", np.random.uniform, 100e-9, 500e-9)
    toolbox.register("attr_fill",    np.random.uniform, 0.1,    0.5)
    toolbox.register("attr_layers",  lambda: int(np.random.randint(1, 21)))
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_lattice, toolbox.attr_fill, toolbox.attr_layers), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_meta(ind):
        """Evaluate metamaterial design for NSGA-II."""
        lattice, fill, layers = ind
        
        # Bounds checking
        if lattice < 100e-9 or lattice > 500e-9:
            return (-1e10, 1000)  # Penalty
        if fill < 0.1 or fill > 0.5:
            return (-1e10, 1000)
        if layers < 1 or layers > 20:
            return (-1e10, 1000)
        
        res = metamaterial_func(lattice, fill, int(layers))
        E = res['total_negative_energy']  # More negative is better energy
        return (E, layers)  # Minimize layers (negative weight in fitness)
    
    toolbox.register("evaluate", eval_meta)
    toolbox.register("mate",    tools.cxBlend, alpha=0.5)
    toolbox.register("mutate",  tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select",  tools.selNSGA2)
    
    def run_nsga2_optimization(population_size=100, generations=50):
        """Run NSGA-II multi-objective optimization."""
        print(f"🧬 Running NSGA-II optimization: {population_size} pop, {generations} gen")
        
        # Initialize population
        pop = toolbox.population(n=population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for gen in range(generations):
            # Select parents
            offspring = toolbox.select(pop, population_size)
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.7:  # Crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < 0.2:  # Mutation probability
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Environmental selection
            pop = toolbox.select(pop + offspring, population_size)
            
            if gen % 10 == 0:
                print(f"   Generation {gen}: Population evolved")
        
        # Extract Pareto front
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        
        print(f"✅ NSGA-II complete: {len(pareto_front)} Pareto-optimal solutions found")
        return pareto_front
    
    def main():
        """Main NSGA-II optimization execution."""
        print("\n🎯 METAMATERIAL PARETO OPTIMIZATION (NSGA-II)")
        print("=" * 60)
        
        # Run optimization
        pareto_solutions = run_nsga2_optimization(population_size=80, generations=40)
        
        # Analyze and display results
        print(f"\n📊 PARETO FRONT ANALYSIS ({len(pareto_solutions)} solutions):")
        print("=" * 60)
        
        results = []
        for i, ind in enumerate(pareto_solutions):
            lat, fill, layers = ind
            res = metamaterial_func(lat, fill, int(layers))
            E = res['total_negative_energy']
            enhancement = res['enhancement_factor']
            
            result_data = {
                'index': i,
                'lattice_nm': lat * 1e9,
                'filling_fraction': fill,
                'layers': int(layers),
                'energy_J': E,
                'enhancement_factor': enhancement,
                'energy_per_layer': E / layers
            }
            results.append(result_data)
            
            print(f"   Solution {i+1:2d}: N={int(layers):2d} layers, "
                  f"a={lat*1e9:5.1f}nm, f={fill:.2f} → "
                  f"E={E:.2e}J, η={enhancement:.1f}x")
        
        # Find key trade-off points
        min_layers = min(results, key=lambda x: x['layers'])
        max_energy = min(results, key=lambda x: x['energy_J'])  # Most negative
        best_efficiency = max(results, key=lambda x: x['energy_per_layer'])
        
        print(f"\n🎯 KEY TRADE-OFF POINTS:")
        print(f"   • Minimum layers: {min_layers['layers']} layers → {min_layers['energy_J']:.2e} J")
        print(f"   • Maximum energy: {max_energy['energy_J']:.2e} J → {max_energy['layers']} layers")
        print(f"   • Best efficiency: {best_efficiency['energy_per_layer']:.2e} J/layer")
        
        # Technology assessment
        fabricable_solutions = [r for r in results if r['lattice_nm'] >= 50 and r['layers'] <= 15]
        print(f"   • Fabricable solutions: {len(fabricable_solutions)}/{len(results)} "
              f"(≥50nm features, ≤15 layers)")
        
        return {
            'pareto_front': pareto_solutions,
            'analysis_results': results,
            'key_points': {
                'min_layers': min_layers,
                'max_energy': max_energy,
                'best_efficiency': best_efficiency
            },
            'fabricable_count': len(fabricable_solutions)
        }

else:
    # Fallback simplified GA
    print("⚠️  Using simplified genetic algorithm fallback")
    
    def run_simplified_ga(population_size=50, generations=20):
        """Simplified GA without DEAP."""
        print(f"🧬 Running simplified GA: {population_size} pop, {generations} gen")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [
                np.random.uniform(100e-9, 500e-9),  # lattice
                np.random.uniform(0.1, 0.5),        # filling
                np.random.randint(1, 21)            # layers
            ]
            population.append(individual)
        
        pareto_archive = []
        
        for gen in range(generations):
            # Evaluate population
            scores = []
            for ind in population:
                lat, fill, layers = ind
                res = metamaterial_func(lat, fill, int(layers))
                energy = res['total_negative_energy']
                scores.append((energy, layers, ind))
            
            # Simple Pareto selection
            for i, (e1, l1, ind1) in enumerate(scores):
                is_dominated = False
                for j, (e2, l2, ind2) in enumerate(scores):
                    if i != j and e2 <= e1 and l2 <= l1 and (e2 < e1 or l2 < l1):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_archive.append(ind1.copy())
            
            # Generate new population
            new_population = []
            for _ in range(population_size):
                if len(pareto_archive) > 0:
                    parent = pareto_archive[np.random.randint(len(pareto_archive))].copy()
                else:
                    parent = population[np.random.randint(population_size)].copy()
                
                # Mutation
                if np.random.random() < 0.3:
                    parent[0] += np.random.normal(0, 50e-9)  # lattice
                    parent[0] = np.clip(parent[0], 100e-9, 500e-9)
                if np.random.random() < 0.3:
                    parent[1] += np.random.normal(0, 0.05)   # filling
                    parent[1] = np.clip(parent[1], 0.1, 0.5)
                if np.random.random() < 0.3:
                    parent[2] += np.random.randint(-2, 3)    # layers
                    parent[2] = np.clip(parent[2], 1, 20)
                
                new_population.append(parent)
            
            population = new_population
            
            if gen % 5 == 0:
                print(f"   Generation {gen}: {len(pareto_archive)} Pareto solutions")
        
        return pareto_archive[:20]  # Return top 20 solutions
    
    def main():
        """Main simplified GA execution."""
        print("\n🎯 METAMATERIAL PARETO OPTIMIZATION (Simplified GA)")
        print("=" * 60)
        
        pareto_solutions = run_simplified_ga()
        
        print(f"\n📊 PARETO SOLUTIONS ({len(pareto_solutions)} found):")
        for i, (lat, fill, layers) in enumerate(pareto_solutions):
            res = metamaterial_func(lat, fill, int(layers))
            E = res['total_negative_energy']
            print(f"   Solution {i+1}: N={int(layers)} layers, "
                  f"a={lat*1e9:.1f}nm, f={fill:.2f} → E={E:.2e}J")
        
        return {'pareto_front': pareto_solutions}

if __name__ == "__main__":
    result = main()
