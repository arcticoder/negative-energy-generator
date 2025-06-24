#!/usr/bin/env python3
"""
High-Dimensional Parameter Scanning for ANEC Optimization
=========================================================

Implements comprehensive parameter space exploration à la GUT Polymerization
to identify viable regions where ANEC < target and violation rates > threshold.

Features:
- Multi-dimensional parameter sweeps across (μ, b, R, τ, exotic_strength, etc.)
- Contour plotting and visualization of viable regions
- Adaptive mesh refinement for efficient exploration
- Statistical analysis of parameter correlations

Author: Negative Energy Generator Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import scipy.optimize as optimize
from scipy.interpolate import griddata
import itertools
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

@dataclass
class ParameterScanConfig:
    """Configuration for high-dimensional parameter scanning."""
    
    # Primary parameters to scan
    polymer_scale_range: Tuple[float, float] = (1e-36, 1e-33)      # μ range
    shape_param_range: Tuple[float, float] = (0.1, 2.0)            # b range
    shell_thickness_range: Tuple[float, float] = (1e-15, 1e-12)    # R range
    redshift_param_range: Tuple[float, float] = (0.01, 1.0)        # τ range
    exotic_strength_range: Tuple[float, float] = (1e-6, 1e-2)      # Δ range
    
    # Secondary parameters
    throat_radius_range: Tuple[float, float] = (1e-16, 1e-13)      # r₀ range
    casimir_separation_range: Tuple[float, float] = (1e-16, 1e-13) # d range
    squeezing_param_range: Tuple[float, float] = (0.5, 5.0)        # r_sq range
    
    # Scan resolution
    grid_resolution: int = 50                    # Points per dimension
    adaptive_refinement_levels: int = 3          # Refinement levels
    refinement_factor: int = 2                   # Refinement resolution factor
    
    # Target criteria
    target_anec: float = -1e5                    # Target ANEC (J·s·m⁻³)
    target_violation_rate: float = 0.30          # Target violation rate
    
    # Computational parameters
    max_evaluations: int = 10000                 # Maximum function evaluations
    parallel_workers: int = None                 # Number of parallel workers
    batch_size: int = 100                        # Batch size for parallel evaluation
    
    # Visualization parameters
    plot_style: str = 'contourf'                 # 'contour', 'contourf', 'scatter'
    colormap: str = 'RdBu_r'                     # Matplotlib colormap
    figure_dpi: int = 150                        # Figure resolution

class HighDimensionalParameterScan:
    """
    High-dimensional parameter space exploration for ANEC optimization.
    
    Systematically explores parameter combinations to identify regions
    of negative ANEC and high violation rates.
    """
    
    def __init__(self, config: ParameterScanConfig = None, evaluation_function: Callable = None):
        self.config = config or ParameterScanConfig()
        self.evaluation_function = evaluation_function
        
        # Set up parallel processing
        if self.config.parallel_workers is None:
            self.config.parallel_workers = max(1, mp.cpu_count() - 1)
        
        # Results storage
        self.scan_results = []
        self.evaluation_count = 0
        self.best_anec = float('inf')
        self.best_parameters = None
        
        print(f"📈 High-Dimensional Parameter Scan Initialized")
        print(f"   Parameters: 8D space (μ, b, R, τ, Δ, r₀, d, r_sq)")
        print(f"   Grid resolution: {self.config.grid_resolution} per dimension")
        print(f"   Target ANEC: {self.config.target_anec:.2e} J·s·m⁻³")
        print(f"   Target violation rate: {self.config.target_violation_rate:.1%}")
        print(f"   Parallel workers: {self.config.parallel_workers}")
    
    def create_parameter_grid(self, dimensions: List[str] = None, 
                            resolution: int = None) -> Dict[str, np.ndarray]:
        """
        Create multi-dimensional parameter grid.
        
        Args:
            dimensions: List of parameter names to include
            resolution: Override default resolution
            
        Returns:
            Dictionary of parameter arrays
        """
        if dimensions is None:
            dimensions = ['polymer_scale', 'shape_param', 'shell_thickness', 
                         'redshift_param', 'exotic_strength']
        
        if resolution is None:
            resolution = self.config.grid_resolution
        
        ranges = {
            'polymer_scale': self.config.polymer_scale_range,
            'shape_param': self.config.shape_param_range,
            'shell_thickness': self.config.shell_thickness_range,
            'redshift_param': self.config.redshift_param_range,
            'exotic_strength': self.config.exotic_strength_range,
            'throat_radius': self.config.throat_radius_range,
            'casimir_separation': self.config.casimir_separation_range,
            'squeezing_param': self.config.squeezing_param_range
        }
        
        grid = {}
        for dim in dimensions:
            if dim in ranges:
                min_val, max_val = ranges[dim]
                if dim in ['polymer_scale', 'shell_thickness', 'throat_radius', 
                          'casimir_separation', 'exotic_strength']:
                    # Log-scale for small parameters
                    grid[dim] = np.logspace(np.log10(min_val), np.log10(max_val), resolution)
                else:
                    # Linear scale for dimensionless parameters
                    grid[dim] = np.linspace(min_val, max_val, resolution)
        
        return grid
    
    def evaluate_parameter_combination(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate ANEC and violation rate for a parameter combination.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Evaluation results
        """
        if self.evaluation_function is None:
            # Dummy evaluation for testing
            return self._dummy_evaluation(params)
        
        try:
            result = self.evaluation_function(params)
            self.evaluation_count += 1
            
            # Track best result
            anec = result.get('anec_total', float('inf'))
            if anec < self.best_anec:
                self.best_anec = anec
                self.best_parameters = params.copy()
            
            return result
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for params {params}: {e}")
            return {
                'anec_total': float('inf'),
                'negative_anec': False,
                'violation_rate': 0.0,
                'evaluation_error': True
            }
    
    def _dummy_evaluation(self, params: Dict[str, float]) -> Dict[str, float]:
        """Dummy evaluation function for testing."""
        mu = params.get('polymer_scale', 1e-35)
        b = params.get('shape_param', 1.0)
        R = params.get('shell_thickness', 1e-14)
        tau = params.get('redshift_param', 0.1)
        delta = params.get('exotic_strength', 1e-3)
        
        # Mock ANEC calculation with some parameter dependencies
        anec = (mu * 1e35)**(-1.5) * np.exp(-b * 3) * (R * 1e14)**0.5 - 1e4
        anec *= (1 + tau) * (delta * 1e3)**0.3
        
        # Add noise
        anec += np.random.normal(0, abs(anec) * 0.1)
        
        violation_rate = np.tanh(abs(anec) / 1e4) * np.random.uniform(0.1, 0.8)
        
        return {
            'anec_total': anec,
            'negative_anec': anec < 0,
            'violation_rate': violation_rate,
            'target_met': anec < self.config.target_anec,
            'violation_target_met': violation_rate > self.config.target_violation_rate
        }
    
    def run_2d_parameter_sweep(self, param1: str, param2: str, 
                             fixed_params: Dict[str, float] = None) -> Dict[str, any]:
        """
        Run 2D parameter sweep for visualization.
        
        Args:
            param1, param2: Parameter names for 2D sweep
            fixed_params: Fixed values for other parameters
            
        Returns:
            2D sweep results with grids and values
        """
        print(f"📊 Running 2D parameter sweep: {param1} vs {param2}")
        
        # Create 2D grid
        grid = self.create_parameter_grid([param1, param2])
        X, Y = np.meshgrid(grid[param1], grid[param2])
        
        # Fixed parameters
        if fixed_params is None:
            fixed_params = {
                'polymer_scale': 1e-35,
                'shape_param': 1.0,
                'shell_thickness': 1e-14,
                'redshift_param': 0.1,
                'exotic_strength': 1e-3,
                'throat_radius': 1e-15,
                'casimir_separation': 5e-15,
                'squeezing_param': 2.0
            }
        
        # Evaluate all combinations
        ANEC = np.zeros_like(X)
        VIOLATION_RATE = np.zeros_like(X)
        NEGATIVE_MASK = np.zeros_like(X, dtype=bool)
        TARGET_MASK = np.zeros_like(X, dtype=bool)
        
        total_points = X.size
        completed = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = fixed_params.copy()
                params[param1] = X[i, j]
                params[param2] = Y[i, j]
                
                result = self.evaluate_parameter_combination(params)
                
                ANEC[i, j] = result['anec_total']
                VIOLATION_RATE[i, j] = result['violation_rate']
                NEGATIVE_MASK[i, j] = result['negative_anec']
                TARGET_MASK[i, j] = result.get('target_met', False)
                
                completed += 1
                if completed % 100 == 0:
                    progress = completed / total_points * 100
                    print(f"   Progress: {progress:.1f}% ({completed}/{total_points})")
        
        # Success statistics
        negative_fraction = NEGATIVE_MASK.sum() / NEGATIVE_MASK.size
        target_fraction = TARGET_MASK.sum() / TARGET_MASK.size
        
        print(f"   ✅ 2D sweep complete!")
        print(f"   Negative ANEC: {negative_fraction:.1%} of parameter space")
        print(f"   Target achieved: {target_fraction:.1%} of parameter space")
        print(f"   Best ANEC: {ANEC.min():.2e} J·s·m⁻³")
        
        return {
            'param1': param1,
            'param2': param2,
            'X': X,
            'Y': Y,
            'ANEC': ANEC,
            'VIOLATION_RATE': VIOLATION_RATE,
            'NEGATIVE_MASK': NEGATIVE_MASK,
            'TARGET_MASK': TARGET_MASK,
            'fixed_params': fixed_params,
            'statistics': {
                'negative_fraction': negative_fraction,
                'target_fraction': target_fraction,
                'best_anec': ANEC.min(),
                'total_evaluations': total_points
            }
        }
    
    def plot_2d_sweep_results(self, sweep_result: Dict[str, any], 
                            save_path: str = None) -> plt.Figure:
        """
        Create visualization of 2D parameter sweep results.
        
        Args:
            sweep_result: Results from run_2d_parameter_sweep
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.figure_dpi)
        fig.suptitle(f'Parameter Sweep: {sweep_result["param1"]} vs {sweep_result["param2"]}', 
                     fontsize=14, fontweight='bold')
        
        X = sweep_result['X']
        Y = sweep_result['Y']
        ANEC = sweep_result['ANEC']
        VIOLATION_RATE = sweep_result['VIOLATION_RATE']
        NEGATIVE_MASK = sweep_result['NEGATIVE_MASK']
        TARGET_MASK = sweep_result['TARGET_MASK']
        
        # 1. ANEC contour plot
        ax1 = axes[0, 0]
        levels = np.linspace(ANEC.min(), ANEC.max(), 20)
        cs1 = ax1.contourf(X, Y, ANEC, levels=levels, cmap=self.config.colormap)
        ax1.contour(X, Y, NEGATIVE_MASK, levels=[0.5], colors='green', 
                   linewidths=2, linestyles='--')
        fig.colorbar(cs1, ax=ax1, label='ANEC (J·s·m⁻³)')
        ax1.set_xlabel(sweep_result['param1'])
        ax1.set_ylabel(sweep_result['param2'])
        ax1.set_title('ANEC Integral')
        ax1.grid(True, alpha=0.3)
        
        # 2. Violation rate plot
        ax2 = axes[0, 1]
        cs2 = ax2.contourf(X, Y, VIOLATION_RATE, levels=20, cmap='viridis')
        ax2.contour(X, Y, VIOLATION_RATE, levels=[self.config.target_violation_rate], 
                   colors='red', linewidths=2, linestyles='-')
        fig.colorbar(cs2, ax=ax2, label='Violation Rate')
        ax2.set_xlabel(sweep_result['param1'])
        ax2.set_ylabel(sweep_result['param2'])
        ax2.set_title('Violation Rate')
        ax2.grid(True, alpha=0.3)
        
        # 3. Success regions
        ax3 = axes[1, 0]
        success_mask = NEGATIVE_MASK & TARGET_MASK
        combined_mask = np.zeros_like(NEGATIVE_MASK, dtype=int)
        combined_mask[NEGATIVE_MASK] = 1
        combined_mask[TARGET_MASK] = 2
        combined_mask[success_mask] = 3
        
        colors = ['white', 'lightblue', 'lightcoral', 'darkgreen']
        cmap = ListedColormap(colors)
        im3 = ax3.imshow(combined_mask, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', cmap=cmap)
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', label='No success'),
            Patch(facecolor='lightblue', label='Negative ANEC'),
            Patch(facecolor='lightcoral', label='Target ANEC'),
            Patch(facecolor='darkgreen', label='Both achieved')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        ax3.set_xlabel(sweep_result['param1'])
        ax3.set_ylabel(sweep_result['param2'])
        ax3.set_title('Success Regions')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = sweep_result['statistics']
        stats_text = f"""
        Scan Statistics:
        
        Total evaluations: {stats['total_evaluations']:,}
        
        Negative ANEC: {stats['negative_fraction']:.1%}
        Target achieved: {stats['target_fraction']:.1%}
        
        Best ANEC: {stats['best_anec']:.2e} J·s·m⁻³
        Target ANEC: {self.config.target_anec:.2e} J·s·m⁻³
        
        Target violation rate: {self.config.target_violation_rate:.1%}
        
        Fixed parameters:
        """
        
        for param, value in sweep_result['fixed_params'].items():
            if param not in [sweep_result['param1'], sweep_result['param2']]:
                stats_text += f"  {param}: {value:.2e}\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            print(f"   💾 Figure saved: {save_path}")
        
        return fig
    
    def run_adaptive_refinement(self, initial_sweep: Dict[str, any], 
                              refinement_levels: int = None) -> Dict[str, any]:
        """
        Run adaptive mesh refinement on promising regions.
        
        Args:
            initial_sweep: Initial 2D sweep results
            refinement_levels: Number of refinement levels
            
        Returns:
            Refined sweep results
        """
        if refinement_levels is None:
            refinement_levels = self.config.adaptive_refinement_levels
        
        print(f"🔍 Running adaptive refinement ({refinement_levels} levels)")
        
        current_result = initial_sweep
        
        for level in range(refinement_levels):
            print(f"   Level {level + 1}/{refinement_levels}")
            
            # Find promising regions (negative ANEC or high violation rate)
            ANEC = current_result['ANEC']
            VIOLATION_RATE = current_result['VIOLATION_RATE']
            X = current_result['X']
            Y = current_result['Y']
            
            # Identify refinement regions
            promising_mask = (ANEC < 0) | (VIOLATION_RATE > self.config.target_violation_rate)
            
            if not promising_mask.any():
                print(f"   No promising regions found at level {level + 1}")
                break
            
            # Find bounding box of promising regions
            indices = np.where(promising_mask)
            i_min, i_max = indices[0].min(), indices[0].max()
            j_min, j_max = indices[1].min(), indices[1].max()
            
            # Expand slightly
            i_min = max(0, i_min - 2)
            i_max = min(X.shape[0] - 1, i_max + 2)
            j_min = max(0, j_min - 2)
            j_max = min(X.shape[1] - 1, j_max + 2)
            
            # Extract refined region bounds
            param1_range = (X[i_min, j_min], X[i_max, j_max])
            param2_range = (Y[i_min, j_min], Y[i_max, j_max])
            
            print(f"      Refining region: {param1_range} × {param2_range}")
            
            # Create refined grid
            resolution = self.config.grid_resolution * self.config.refinement_factor
            param1_refined = np.linspace(param1_range[0], param1_range[1], resolution)
            param2_refined = np.linspace(param2_range[0], param2_range[1], resolution)
            X_refined, Y_refined = np.meshgrid(param1_refined, param2_refined)
            
            # Evaluate refined grid
            ANEC_refined = np.zeros_like(X_refined)
            VIOLATION_RATE_refined = np.zeros_like(X_refined)
            NEGATIVE_MASK_refined = np.zeros_like(X_refined, dtype=bool)
            TARGET_MASK_refined = np.zeros_like(X_refined, dtype=bool)
            
            for i in range(X_refined.shape[0]):
                for j in range(X_refined.shape[1]):
                    params = current_result['fixed_params'].copy()
                    params[current_result['param1']] = X_refined[i, j]
                    params[current_result['param2']] = Y_refined[i, j]
                    
                    result = self.evaluate_parameter_combination(params)
                    
                    ANEC_refined[i, j] = result['anec_total']
                    VIOLATION_RATE_refined[i, j] = result['violation_rate']
                    NEGATIVE_MASK_refined[i, j] = result['negative_anec']
                    TARGET_MASK_refined[i, j] = result.get('target_met', False)
            
            # Update current result
            current_result = {
                'param1': current_result['param1'],
                'param2': current_result['param2'],
                'X': X_refined,
                'Y': Y_refined,
                'ANEC': ANEC_refined,
                'VIOLATION_RATE': VIOLATION_RATE_refined,
                'NEGATIVE_MASK': NEGATIVE_MASK_refined,
                'TARGET_MASK': TARGET_MASK_refined,
                'fixed_params': current_result['fixed_params'],
                'refinement_level': level + 1
            }
            
            # Update statistics
            negative_fraction = NEGATIVE_MASK_refined.sum() / NEGATIVE_MASK_refined.size
            target_fraction = TARGET_MASK_refined.sum() / TARGET_MASK_refined.size
            
            print(f"      Negative ANEC: {negative_fraction:.1%}")
            print(f"      Target achieved: {target_fraction:.1%}")
            print(f"      Best ANEC: {ANEC_refined.min():.2e}")
        
        print(f"   ✅ Adaptive refinement complete!")
        return current_result

def demo_parameter_scanning():
    """Demonstrate high-dimensional parameter scanning."""
    print("📈 High-Dimensional Parameter Scanning Demo")
    print("=" * 60)
    
    # Create scanner
    config = ParameterScanConfig(
        grid_resolution=30,
        target_anec=-1e4,
        target_violation_rate=0.25
    )
    
    scanner = HighDimensionalParameterScan(config)
    
    # Run 2D sweep (μ vs b)
    sweep_result = scanner.run_2d_parameter_sweep(
        'polymer_scale', 'shape_param',
        fixed_params={
            'shell_thickness': 5e-14,
            'redshift_param': 0.2,
            'exotic_strength': 2e-3
        }
    )
    
    # Create visualization
    fig = scanner.plot_2d_sweep_results(sweep_result, save_path='parameter_sweep_demo.png')
    
    # Run adaptive refinement
    refined_result = scanner.run_adaptive_refinement(sweep_result, refinement_levels=2)
    
    return scanner, sweep_result, refined_result

if __name__ == "__main__":
    demo_parameter_scanning()
