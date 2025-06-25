# File: src/analysis/meta_jpa_pareto_plot.py
"""
Joint Pareto Analysis for Metamaterial and JPA Optimization

Combines metamaterial energy optimization with JPA squeezing to create
joint trade-off plots for multi-objective system design.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.analysis.meta_pareto_ga import main as meta_pareto_main
    META_AVAILABLE = True
except ImportError:
    META_AVAILABLE = False
    print("⚠️  meta_pareto_ga not available")

try:
    from src.analysis.jpa_bayes_opt import main as jpa_bayes_main
    JPA_AVAILABLE = True
except ImportError:
    JPA_AVAILABLE = False
    print("⚠️  jpa_bayes_opt not available")

try:
    from src.analysis.in_silico_stack_and_squeeze import (
        simulate_photonic_metamaterial, 
        simulate_jpa_squeezed_vacuum
    )
    INSILICO_AVAILABLE = True
except ImportError:
    INSILICO_AVAILABLE = False
    print("⚠️  in_silico_stack_and_squeeze not available - using fallbacks")

# Fallback functions if modules unavailable
def simulate_metamaterial_fallback(lattice, fill, layers):
    """Fallback metamaterial simulation."""
    base_energy = -1e-15
    enhancement = np.sqrt(layers) * (1 - abs(lattice - 250e-9) / 250e-9) * (1 - abs(fill - 0.35) / 0.35)
    return {'total_negative_energy': base_energy * enhancement}

def simulate_jpa_fallback(freq, pump, temp):
    """Fallback JPA simulation."""
    optimal_pump = 0.15
    efficiency = 1 / (1 + 10 * (pump - optimal_pump)**2)
    r = 2.0 * efficiency * 0.9  # Thermal factor approximation
    squeezing_db = 20 * np.log10(np.exp(-r)) if r > 0 else 0
    energy = -np.sinh(r)**2 * 1.054e-34 * freq * 1e-18
    return {'squeezing_db': squeezing_db, 'total_energy': energy}

# Select available functions
if INSILICO_AVAILABLE:
    meta_func = simulate_photonic_metamaterial
    jpa_func = simulate_jpa_squeezed_vacuum
else:
    meta_func = simulate_metamaterial_fallback
    jpa_func = simulate_jpa_fallback

def generate_metamaterial_pareto_data():
    """Generate metamaterial Pareto front data."""
    print("🧬 Generating metamaterial Pareto data...")
    
    if META_AVAILABLE:
        try:
            # Use real Pareto optimization
            result = meta_pareto_main()
            if 'pareto_front' in result:
                pareto_solutions = result['pareto_front']
                
                pareto_data = []
                for ind in pareto_solutions:
                    if hasattr(ind, '__iter__') and len(ind) >= 3:
                        lat, fill, layers = ind[:3]
                        res = meta_func(lat, fill, int(layers))
                        pareto_data.append({
                            'lattice': lat,
                            'filling': fill,
                            'layers': int(layers),
                            'energy': res['total_negative_energy'],
                            'energy_magnitude': abs(res['total_negative_energy'])
                        })
                
                print(f"   ✅ Generated {len(pareto_data)} Pareto solutions")
                return pareto_data
        except Exception as e:
            print(f"   ⚠️  Pareto generation failed: {e}")
    
    # Fallback: generate sample designs
    print("   🔄 Using fallback metamaterial designs")
    designs = []
    
    # Sample different layer counts and configurations
    for layers in [5, 8, 10, 12, 15, 20]:
        for lattice in [200e-9, 250e-9, 300e-9, 400e-9]:
            for fill in [0.25, 0.35, 0.45]:
                res = meta_func(lattice, fill, layers)
                designs.append({
                    'lattice': lattice,
                    'filling': fill,
                    'layers': layers,
                    'energy': res['total_negative_energy'],
                    'energy_magnitude': abs(res['total_negative_energy'])
                })
    
    # Simple Pareto filtering
    pareto_data = []
    for i, design1 in enumerate(designs):
        is_dominated = False
        for j, design2 in enumerate(designs):
            if i != j:
                # Check if design2 dominates design1
                if (design2['energy'] <= design1['energy'] and 
                    design2['layers'] <= design1['layers'] and
                    (design2['energy'] < design1['energy'] or design2['layers'] < design1['layers'])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_data.append(design1)
    
    print(f"   ✅ Generated {len(pareto_data)} Pareto-filtered designs")
    return pareto_data

def generate_jpa_optimization_curve():
    """Generate JPA optimization curve data."""
    print("⚡ Generating JPA optimization curve...")
    
    if JPA_AVAILABLE:
        try:
            # Use real Bayesian optimization
            result = jpa_bayes_main()
            if 'optimization_result' in result:
                opt_params = result['optimization_result']['optimal_params']
                optimal_eps = opt_params['epsilon']
                optimal_delta = opt_params['delta_normalized']
                
                print(f"   ✅ Found optimal: ε={optimal_eps:.3f}, δ={optimal_delta:.3f}")
                
                # Generate curve around optimal point
                eps_range = np.linspace(0.01, 0.3, 25)
                jpa_data = []
                
                for eps in eps_range:
                    # Use optimal delta for curve
                    pump_eff = eps * (1 - 0.5 * abs(optimal_delta))
                    pump_eff = np.clip(pump_eff, 0.01, 0.3)
                    
                    res = jpa_func(6e9, pump_eff, 0.015)
                    jpa_data.append({
                        'epsilon': eps,
                        'pump_effective': pump_eff,
                        'squeezing_db': res['squeezing_db'],
                        'energy': res['total_energy'],
                        'energy_magnitude': abs(res['total_energy'])
                    })
                
                return jpa_data, optimal_eps
        except Exception as e:
            print(f"   ⚠️  JPA optimization failed: {e}")
    
    # Fallback: parameter sweep
    print("   🔄 Using fallback JPA parameter sweep")
    eps_vals = np.linspace(0.01, 0.3, 20)
    jpa_data = []
    best_db = -float('inf')
    optimal_eps = 0.15
    
    for eps in eps_vals:
        res = jpa_func(6e9, eps, 0.015)
        jpa_data.append({
            'epsilon': eps,
            'pump_effective': eps,
            'squeezing_db': res['squeezing_db'],
            'energy': res['total_energy'],
            'energy_magnitude': abs(res['total_energy'])
        })
        
        if res['squeezing_db'] > best_db:
            best_db = res['squeezing_db']
            optimal_eps = eps
    
    print(f"   ✅ Generated {len(jpa_data)} JPA points, best: ε={optimal_eps:.3f}")
    return jpa_data, optimal_eps

def create_joint_pareto_plot():
    """Create joint Pareto plot combining metamaterial and JPA results."""
    print("\n📊 JOINT METAMATERIAL-JPA PARETO ANALYSIS")
    print("=" * 60)
    
    # Generate data
    meta_data = generate_metamaterial_pareto_data()
    jpa_data, optimal_jpa_eps = generate_jpa_optimization_curve()
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Joint Metamaterial-JPA Optimization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Metamaterial Energy vs Layers (Pareto front)
    if meta_data:
        energies = [d['energy_magnitude'] for d in meta_data]
        layers = [d['layers'] for d in meta_data]
        
        ax1.scatter(layers, energies, c='red', alpha=0.7, s=60, label='Pareto Front')
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('|Negative Energy| (J)')
        ax1.set_title('Metamaterial: Energy vs Complexity')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Annotate key points
        if len(meta_data) > 0:
            min_layers_point = min(meta_data, key=lambda x: x['layers'])
            max_energy_point = max(meta_data, key=lambda x: x['energy_magnitude'])
            
            ax1.annotate(f'Min Layers\n({min_layers_point["layers"]}L)', 
                        xy=(min_layers_point['layers'], min_layers_point['energy_magnitude']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax1.annotate(f'Max Energy\n({max_energy_point["energy_magnitude"]:.1e}J)', 
                        xy=(max_energy_point['layers'], max_energy_point['energy_magnitude']),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: JPA Squeezing vs Pump Power
    if jpa_data:
        epsilons = [d['epsilon'] for d in jpa_data]
        squeezings = [d['squeezing_db'] for d in jpa_data]
        
        ax2.plot(epsilons, squeezings, 'b-o', linewidth=2, markersize=4, label='Squeezing Curve')
        ax2.axvline(x=optimal_jpa_eps, color='red', linestyle='--', alpha=0.7, label=f'Optimal ε={optimal_jpa_eps:.3f}')
        ax2.set_xlabel('Pump Amplitude ε')
        ax2.set_ylabel('Squeezing (dB)')
        ax2.set_title('JPA: Squeezing vs Pump Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Find and annotate maximum
        max_squeeze_point = max(jpa_data, key=lambda x: x['squeezing_db'])
        ax2.annotate(f'Max: {max_squeeze_point["squeezing_db"]:.1f}dB', 
                    xy=(max_squeeze_point['epsilon'], max_squeeze_point['squeezing_db']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 3: Joint Energy Comparison
    if meta_data and jpa_data:
        meta_energies = [d['energy_magnitude'] for d in meta_data]
        jpa_energies = [d['energy_magnitude'] for d in jpa_data]
        
        ax3.hist(meta_energies, bins=10, alpha=0.6, color='red', label='Metamaterial', density=True)
        ax3.hist(jpa_energies, bins=10, alpha=0.6, color='blue', label='JPA', density=True)
        ax3.set_xlabel('|Negative Energy| (J)')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Energy Distribution Comparison')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined Performance Map
    if meta_data and jpa_data:
        # Create combined performance metrics
        
        # Best metamaterial energy
        best_meta = max(meta_data, key=lambda x: x['energy_magnitude'])
        meta_best_energy = best_meta['energy_magnitude']
        meta_best_layers = best_meta['layers']
        
        # Best JPA squeezing
        best_jpa = max(jpa_data, key=lambda x: x['squeezing_db'])
        jpa_best_energy = best_jpa['energy_magnitude']
        jpa_best_squeezing = best_jpa['squeezing_db']
        
        # Performance comparison
        categories = ['Energy\n(Metamaterial)', 'Energy\n(JPA)', 'Squeezing\n(JPA)', 'Complexity\n(Meta Layers)']
        values = [
            meta_best_energy / 1e-16,  # Normalized to 10^-16 J
            jpa_best_energy / 1e-16,
            jpa_best_squeezing,
            meta_best_layers
        ]
        colors = ['red', 'blue', 'cyan', 'orange']
        
        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Normalized Performance')
        ax4.set_title('Best Performance Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'meta_jpa_joint_pareto_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved as: {plot_filename}")
    
    # Analysis summary
    print(f"\n🎯 JOINT ANALYSIS SUMMARY:")
    if meta_data:
        print(f"   • Metamaterial Pareto solutions: {len(meta_data)}")
        best_meta = max(meta_data, key=lambda x: x['energy_magnitude'])
        print(f"   • Best metamaterial energy: {best_meta['energy_magnitude']:.2e} J ({best_meta['layers']} layers)")
    
    if jpa_data:
        print(f"   • JPA optimization points: {len(jpa_data)}")
        best_jpa = max(jpa_data, key=lambda x: x['squeezing_db'])
        print(f"   • Best JPA squeezing: {best_jpa['squeezing_db']:.1f} dB (ε={best_jpa['epsilon']:.3f})")
    
    # Technology recommendations
    print(f"\n📋 TECHNOLOGY RECOMMENDATIONS:")
    if meta_data and jpa_data:
        # Find balanced solutions
        balanced_meta = [d for d in meta_data if d['layers'] <= 12 and d['energy_magnitude'] > 1e-16]
        high_squeeze_jpa = [d for d in jpa_data if d['squeezing_db'] > 10]
        
        print(f"   • Balanced metamaterial designs: {len(balanced_meta)} (≤12 layers, >1e-16 J)")
        print(f"   • High-squeezing JPA configs: {len(high_squeeze_jpa)} (>10 dB)")
        
        if balanced_meta and high_squeeze_jpa:
            print("   ✅ Hybrid approach recommended: Moderate metamaterial + optimized JPA")
        elif len(meta_data) > len(jpa_data):
            print("   📈 Focus on metamaterial optimization")
        else:
            print("   ⚡ Focus on JPA optimization")
    
    plt.show()
    
    return {
        'metamaterial_data': meta_data,
        'jpa_data': jpa_data,
        'plot_filename': plot_filename
    }

def main():
    """Main joint analysis execution."""
    print("\n🎯 JOINT METAMATERIAL-JPA PARETO OPTIMIZATION")
    print("=" * 70)
    
    result = create_joint_pareto_plot()
    
    print("\n✅ Joint Pareto analysis complete!")
    print("   📊 Multi-objective trade-offs visualized")
    print("   🎯 Technology recommendations provided")
    print("   💾 Results saved for further analysis")
    
    return result

if __name__ == "__main__":
    result = main()
