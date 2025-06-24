"""
Comprehensive Validation Driver

Integrates high-resolution simulations, radiative corrections, and quantum-interest 
optimization to provide complete theoretical model validation.
"""

import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .high_resolution_sims import (
    run_parameter_sweep, find_sweet_spots, plot_parameter_space, 
    save_results as save_sim_results
)
from .radiative_corrections import (
    analyze_loop_convergence, loop_correction_parameter_sweep, 
    validate_anec_robustness
)
from .quantum_interest import (
    analyze_warp_bubble_quantum_interest, quantum_interest_parameter_sweep,
    plot_quantum_interest_analysis
)

logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """Main validation framework integrating all theoretical refinements."""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_full_validation(self, 
                          mu_center: float = 0.095, mu_width: float = 0.008,
                          R_center: float = 2.3, R_width: float = 0.2,
                          tau_center: float = 1.2, tau_width: float = 0.15,
                          n_points: int = 10) -> Dict:
        """
        Execute complete validation pipeline:
        1. High-resolution parameter sweeps
        2. Radiative corrections analysis  
        3. Quantum-interest optimization
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting comprehensive validation at {timestamp}")
        
        # Phase 1: High-resolution simulations
        logger.info("Phase 1: High-resolution parameter sweep")
        sim_results = run_parameter_sweep(
            mu_center, mu_width, R_center, R_width, tau_center, tau_width, n_points
        )
        
        sweet_spots = find_sweet_spots(sim_results, min_violation_rate=70.0, min_negative_energy=-1e4)
        
        # Save simulation results
        sim_file = self.output_dir / f"parameter_sweep_{timestamp}.json"
        save_sim_results(sim_results, str(sim_file))
        
        # Generate plots
        plot_file = self.output_dir / f"parameter_space_{timestamp}.png"
        plot_parameter_space(sim_results, str(plot_file))
        
        logger.info(f"Found {len(sweet_spots)} sweet spots from {len(sim_results)} simulations")
        
        # Phase 2: Radiative corrections for top sweet spots
        logger.info("Phase 2: Radiative corrections analysis")
        top_spots = sweet_spots[:min(5, len(sweet_spots))]  # Analyze top 5 spots
        
        loop_results = []
        anec_robustness = []
        
        for spot in top_spots:
            params = spot.params
            
            # Analyze loop corrections
            loop_correction = analyze_loop_convergence(params.mu, params.R, params.tau)
            loop_results.append({
                'mu': params.mu, 'R': params.R, 'tau': params.tau,
                'tree_level': loop_correction.tree_level,
                'one_loop': loop_correction.one_loop,
                'two_loop': loop_correction.two_loop,
                'total': loop_correction.total,
                'convergent': loop_correction.convergent
            })
            
            # Validate ANEC robustness (using target ANEC from barrier assessment)
            target_anec = -3.58e5  # J·s·m⁻³
            robustness = validate_anec_robustness(target_anec, loop_correction)
            anec_robustness.append({
                'params': {'mu': params.mu, 'R': params.R, 'tau': params.tau},
                'robust': robustness['robust'],
                'relative_change': robustness['relative_change'],
                'corrected_anec': robustness['corrected_anec']
            })
        
        # Phase 3: Quantum-interest optimization
        logger.info("Phase 3: Quantum-interest trade-off analysis")
        qi_results = []
        
        for spot in top_spots:
            params = spot.params
            qi_analysis = analyze_warp_bubble_quantum_interest(
                params.mu, params.R, params.tau, characteristic_energy=abs(spot.I_neg)
            )
            qi_results.append(qi_analysis)
        
        # Compile comprehensive results
        self.results = {
            'timestamp': timestamp,
            'parameters': {
                'mu_center': mu_center, 'mu_width': mu_width,
                'R_center': R_center, 'R_width': R_width, 
                'tau_center': tau_center, 'tau_width': tau_width,
                'n_points': n_points
            },
            'simulation_results': {
                'total_simulations': len(sim_results),
                'sweet_spots': len(sweet_spots),
                'top_violations': [
                    {
                        'mu': s.params.mu, 'R': s.params.R, 'tau': s.params.tau,
                        'violation_rate': s.violation_rate, 'I_neg': s.I_neg, 'stable': s.stable
                    } for s in sweet_spots[:10]
                ]
            },
            'radiative_corrections': {
                'loop_analysis': loop_results,
                'anec_robustness': anec_robustness,
                'convergent_fraction': sum(r['convergent'] for r in loop_results) / len(loop_results)
            },
            'quantum_interest': {
                'analyses': qi_results,
                'successful_optimizations': sum(1 for qi in qi_results if qi.get('success', True))
            }
        }
        
        # Save comprehensive results
        results_file = self.output_dir / f"comprehensive_validation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive validation completed. Results saved to {results_file}")
        
        return self.results
        
    def generate_summary_report(self) -> str:
        """Generate human-readable summary of validation results."""
        if not self.results:
            return "No validation results available. Run validation first."
        
        report_lines = [
            "COMPREHENSIVE THEORETICAL VALIDATION REPORT",
            "=" * 50,
            f"Generated: {self.results['timestamp']}",
            "",
            "1. HIGH-RESOLUTION PARAMETER SWEEP",
            "-" * 35,
            f"Total simulations: {self.results['simulation_results']['total_simulations']}",
            f"Stable sweet spots: {self.results['simulation_results']['sweet_spots']}",
            "",
            "Top performing parameter combinations:",
        ]
        
        for i, spot in enumerate(self.results['simulation_results']['top_violations'][:5]):
            report_lines.append(
                f"  {i+1}. μ={spot['mu']:.4f}, R={spot['R']:.3f}, τ={spot['tau']:.3f} "
                f"(violation: {spot['violation_rate']:.1f}%, stable: {spot['stable']})"
            )
        
        report_lines.extend([
            "",
            "2. RADIATIVE CORRECTIONS ANALYSIS", 
            "-" * 35,
            f"Convergent series: {self.results['radiative_corrections']['convergent_fraction']:.1%}",
            "",
            "ANEC robustness under quantum corrections:"
        ])
        
        for robustness in self.results['radiative_corrections']['anec_robustness']:
            params = robustness['params']
            report_lines.append(
                f"  μ={params['mu']:.4f}, R={params['R']:.3f}, τ={params['tau']:.3f}: "
                f"{'ROBUST' if robustness['robust'] else 'NOT ROBUST'} "
                f"(change: {robustness['relative_change']:.1%})"
            )
        
        report_lines.extend([
            "",
            "3. QUANTUM-INTEREST OPTIMIZATION",
            "-" * 35,
            f"Successful optimizations: {self.results['quantum_interest']['successful_optimizations']}",
            "",
            "Energy efficiency analysis:"
        ])
        
        for qi in self.results['quantum_interest']['analyses']:
            if qi.get('simple_optimization'):
                opt = qi['simple_optimization']
                warp = qi['warp_params']
                report_lines.append(
                    f"  μ={warp['mu']:.4f}, R={warp['R']:.3f}, τ={warp['tau']:.3f}: "
                    f"efficiency={opt.efficiency:.3f}, net_cost={opt.net_energy:.2e}"
                )
        
        report_lines.extend([
            "",
            "OVERALL ASSESSMENT",
            "-" * 18,
            self._generate_overall_assessment()
        ])
        
        return "\n".join(report_lines)
    
    def _generate_overall_assessment(self) -> str:
        """Generate overall assessment of theoretical validity."""
        sim_success = self.results['simulation_results']['sweet_spots'] > 0
        loop_success = self.results['radiative_corrections']['convergent_fraction'] > 0.8
        qi_success = self.results['quantum_interest']['successful_optimizations'] > 0
        anec_robust = any(r['robust'] for r in self.results['radiative_corrections']['anec_robustness'])
        
        if sim_success and loop_success and qi_success and anec_robust:
            return ("✅ THEORETICAL MODEL VALIDATED\n"
                   "   All three validation phases successful.\n"
                   "   Ready for experimental implementation.")
        elif sim_success and (loop_success or anec_robust):
            return ("⚠️  PARTIAL VALIDATION\n"
                   "   Core results solid but some refinements needed.\n"
                   "   Proceed with caution to experimental phase.")
        else:
            return ("❌ VALIDATION CONCERNS\n"
                   "   Significant theoretical issues identified.\n"
                   "   Further theoretical work required.")
    
    def save_report(self, filename: Optional[str] = None):
        """Save validation report to file."""
        if not filename:
            timestamp = self.results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = f"validation_report_{timestamp}.txt"
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(self.generate_summary_report())
        
        logger.info(f"Validation report saved to {report_path}")


def run_validation_pipeline(quick_mode: bool = False) -> ComprehensiveValidator:
    """
    Convenience function to run the complete validation pipeline.
    
    Args:
        quick_mode: If True, use smaller parameter grids for faster execution
    """
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    validator = ComprehensiveValidator()
    
    # Adjust parameters for quick vs thorough mode
    n_points = 8 if quick_mode else 12
    
    print("Starting comprehensive theoretical validation...")
    print(f"Mode: {'Quick' if quick_mode else 'Thorough'}")
    print(f"Parameter grid: {n_points}³ = {n_points**3} points")
    
    results = validator.run_full_validation(n_points=n_points)
    
    print("\n" + "="*60)
    print(validator.generate_summary_report())
    print("="*60)
    
    validator.save_report()
    
    return validator


if __name__ == "__main__":
    # Run validation pipeline
    import sys
    
    quick = "--quick" in sys.argv
    validator = run_validation_pipeline(quick_mode=quick)
    
    print(f"\nValidation complete! Results saved to: {validator.output_dir}")
    print("Check the generated report and plots for detailed analysis.")
