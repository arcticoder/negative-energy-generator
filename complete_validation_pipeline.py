"""
Complete Validation Pipeline: Tree + Loop + Quantum Interest Optimization

This script integrates all advanced theoretical modules to achieve:
- ANEC violations: ⟨∫T₀₀⟩ ≲ −10⁵ J·s·m⁻³
- High violation rates: ≥50–75% of spacetime
- Ford-Roman factor violations: 10³–10⁴×
- Optimized quantum-interest trade-offs

Run this after theory-level targets are validated before hardware prototyping.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our modules
from src.validation.high_res_sweep import HighResolutionSimulation, AdvancedStabilityAnalyzer
from src.corrections.radiative import RadiativeCorrections
from src.validation.quantum_interest import (
    optimize_quantum_interest_simple, 
    analyze_warp_bubble_quantum_interest,
    quantum_interest_parameter_sweep,
    plot_quantum_interest_analysis
)
from src.quantum.field_algebra import PolymerFieldAlgebra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'validation_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteValidationPipeline:
    """
    Complete validation pipeline integrating all theoretical modules.
    """
    
    def __init__(self, output_dir: str = "validation_results"):
        """Initialize the validation pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize modules
        self.high_res_sim = HighResolutionSimulation()
        self.stability_analyzer = AdvancedStabilityAnalyzer()
        self.radiative = RadiativeCorrections(mass=0.0, coupling=1.0, cutoff=100.0)
        self.polymer_algebra = PolymerFieldAlgebra(gamma=0.2375)
        
        # Results storage
        self.results = {
            'tree_level': {},
            'one_loop': {},
            'two_loop': {},
            'quantum_interest': {},
            'summary': {}
        }
        
        logger.info(f"Validation pipeline initialized. Output: {self.output_dir}")
    
    def run_tree_level_optimization(self, target_anec: float = -1e5) -> Dict:
        """
        Run tree-level parameter optimization targeting specific ANEC violation.
        """
        logger.info("Starting tree-level optimization...")
        
        # High-resolution parameter sweep
        mu_range = (0.01, 0.2, 100)  # (min, max, points)
        R_range = (0.5, 5.0, 100)
        tau_range = (0.1, 3.0, 100)
        
        best_params = None
        best_anec = 0
        violation_rates = []
        ford_roman_factors = []
        
        total_points = mu_range[2] * R_range[2] * tau_range[2]
        logger.info(f"Scanning {total_points} parameter combinations...")
        
        count = 0
        for mu in np.linspace(mu_range[0], mu_range[1], mu_range[2]):
            for R in np.linspace(R_range[0], R_range[1], R_range[2]):
                for tau in np.linspace(tau_range[0], tau_range[1], tau_range[2]):
                    count += 1
                    if count % 10000 == 0:
                        logger.info(f"Progress: {count}/{total_points} ({100*count/total_points:.1f}%)")
                    
                    try:
                        # Run high-resolution simulation
                        result = self.high_res_sim.comprehensive_anec_analysis(
                            mu=mu, R=R, tau=tau, 
                            r_points=200, t_points=200
                        )
                        
                        anec_integral = result['anec_integral']
                        violation_rate = result.get('violation_rate', 0)
                        ford_roman = result.get('ford_roman_factor', 0)
                        
                        # Check if this beats our target
                        if anec_integral < target_anec and anec_integral < best_anec:
                            best_anec = anec_integral
                            best_params = {'mu': mu, 'R': R, 'tau': tau}
                            logger.info(f"New best: ANEC = {anec_integral:.2e}, params = {best_params}")
                        
                        # Store statistics
                        if anec_integral < 0:
                            violation_rates.append(violation_rate)
                            ford_roman_factors.append(ford_roman)
                    
                    except Exception as e:
                        logger.warning(f"Failed at mu={mu:.3f}, R={R:.3f}, tau={tau:.3f}: {e}")
                        continue
        
        # Analyze violation statistics
        if violation_rates:
            mean_violation_rate = np.mean(violation_rates)
            mean_ford_roman = np.mean(ford_roman_factors)
            max_ford_roman = np.max(ford_roman_factors)
        else:
            mean_violation_rate = 0
            mean_ford_roman = 0
            max_ford_roman = 0
        
        tree_results = {
            'best_params': best_params,
            'best_anec': best_anec,
            'target_achieved': best_anec < target_anec,
            'mean_violation_rate': mean_violation_rate,
            'mean_ford_roman': mean_ford_roman,
            'max_ford_roman': max_ford_roman,
            'total_tested': count
        }
        
        self.results['tree_level'] = tree_results
        logger.info(f"Tree-level optimization complete: {tree_results}")
        return tree_results
    
    def compute_radiative_corrections(self, params: Dict) -> Dict:
        """
        Compute 1-loop and 2-loop corrections for given parameters.
        """
        if not params:
            logger.warning("No parameters provided for radiative corrections")
            return {}
        
        logger.info(f"Computing radiative corrections for: {params}")
        
        mu, R, tau = params['mu'], params['R'], params['tau']
        
        try:
            # One-loop corrections
            one_loop_result = self.radiative.one_loop_correction(R=R, tau=tau)
            
            # Two-loop corrections
            two_loop_result = self.radiative.two_loop_correction(R=R, tau=tau)
            
            # Combine corrections
            total_correction = one_loop_result['T00_correction'] + two_loop_result['T00_correction']
            
            corrections = {
                'one_loop': one_loop_result,
                'two_loop': two_loop_result,
                'total_correction': total_correction,
                'params': params
            }
            
            self.results['one_loop'] = one_loop_result
            self.results['two_loop'] = two_loop_result
            
            logger.info(f"Radiative corrections: 1-loop = {one_loop_result['T00_correction']:.2e}, "
                       f"2-loop = {two_loop_result['T00_correction']:.2e}")
            
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to compute radiative corrections: {e}")
            return {}
    
    def optimize_quantum_interest(self, params: Dict, anec_magnitude: float) -> Dict:
        """
        Optimize quantum interest for given warp bubble parameters.
        """
        if not params:
            logger.warning("No parameters for quantum interest optimization")
            return {}
        
        logger.info(f"Optimizing quantum interest for ANEC magnitude: {anec_magnitude:.2e}")
        
        try:
            # Analyze quantum interest for warp bubble
            qi_analysis = analyze_warp_bubble_quantum_interest(
                mu=params['mu'], 
                R=params['R'], 
                tau=params['tau'],
                characteristic_energy=abs(anec_magnitude)
            )
            
            # Parameter sweep for efficiency optimization
            qi_sweep = quantum_interest_parameter_sweep(
                A_minus_range=(abs(anec_magnitude)*0.1, abs(anec_magnitude)*2.0),
                sigma_range=(params['tau']*0.5, params['tau']*2.0),
                n_points=50
            )
            
            if qi_sweep:
                # Find most efficient configuration
                efficiencies = [pulse.efficiency for pulse in qi_sweep]
                best_idx = np.argmax(efficiencies)
                best_pulse = qi_sweep[best_idx]
                
                qi_results = {
                    'analysis': qi_analysis,
                    'sweep_results': qi_sweep,
                    'best_efficiency': best_pulse.efficiency,
                    'best_net_energy': best_pulse.net_energy,
                    'best_delay': best_pulse.delay,
                    'mean_efficiency': np.mean(efficiencies),
                    'efficiency_std': np.std(efficiencies)
                }
                
                # Save efficiency plot
                plot_path = self.output_dir / "quantum_interest_analysis.png"
                plot_quantum_interest_analysis(qi_sweep, str(plot_path))
                
                self.results['quantum_interest'] = qi_results
                
                logger.info(f"QI optimization: Best efficiency = {best_pulse.efficiency:.3f}, "
                           f"Mean = {np.mean(efficiencies):.3f}")
                
                return qi_results
            else:
                logger.warning("Quantum interest sweep returned no results")
                return {}
                
        except Exception as e:
            logger.error(f"Failed quantum interest optimization: {e}")
            return {}
    
    def analyze_corrected_anec(self, tree_anec: float, corrections: Dict) -> Dict:
        """
        Analyze ANEC after including radiative corrections.
        """
        if not corrections:
            return {'corrected_anec': tree_anec, 'correction_impact': 0}
        
        corrected_anec = tree_anec + corrections.get('total_correction', 0)
        correction_impact = abs(corrections.get('total_correction', 0)) / abs(tree_anec) if tree_anec != 0 else 0
        
        analysis = {
            'tree_anec': tree_anec,
            'corrected_anec': corrected_anec,
            'total_correction': corrections.get('total_correction', 0),
            'correction_impact': correction_impact,
            'anec_sign_preserved': (tree_anec < 0) == (corrected_anec < 0),
            'violation_robust': corrected_anec < -1e4  # Still significantly negative
        }
        
        logger.info(f"ANEC analysis: Tree = {tree_anec:.2e}, "
                   f"Corrected = {corrected_anec:.2e}, "
                   f"Impact = {correction_impact:.1%}")
        
        return analysis
    
    def run_complete_validation(self) -> Dict:
        """
        Run the complete validation pipeline.
        """
        logger.info("="*80)
        logger.info("STARTING COMPLETE VALIDATION PIPELINE")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Step 1: Tree-level optimization
        logger.info("STEP 1: Tree-level parameter optimization")
        tree_results = self.run_tree_level_optimization(target_anec=-1e5)
        
        # Step 2: Radiative corrections
        logger.info("STEP 2: Computing radiative corrections")
        corrections = {}
        if tree_results.get('best_params'):
            corrections = self.compute_radiative_corrections(tree_results['best_params'])
        
        # Step 3: Corrected ANEC analysis
        logger.info("STEP 3: Analyzing corrected ANEC")
        anec_analysis = self.analyze_corrected_anec(
            tree_results.get('best_anec', 0), 
            corrections
        )
        
        # Step 4: Quantum interest optimization
        logger.info("STEP 4: Quantum interest optimization")
        qi_results = {}
        if tree_results.get('best_params'):
            qi_results = self.optimize_quantum_interest(
                tree_results['best_params'],
                abs(anec_analysis.get('corrected_anec', tree_results.get('best_anec', 1e5)))
            )
        
        # Step 5: Final validation check
        logger.info("STEP 5: Final validation assessment")
        validation_summary = self.generate_validation_summary(
            tree_results, corrections, anec_analysis, qi_results
        )
        
        # Save complete results
        self.save_results(validation_summary)
        
        end_time = datetime.now()
        logger.info(f"Complete validation finished in {end_time - start_time}")
        
        return validation_summary
    
    def generate_validation_summary(self, tree_results: Dict, corrections: Dict, 
                                  anec_analysis: Dict, qi_results: Dict) -> Dict:
        """
        Generate final validation summary and assessment.
        """
        # Target criteria
        targets = {
            'anec_magnitude': 1e5,      # |ANEC| ≥ 10⁵ J·s·m⁻³
            'violation_rate': 0.5,      # ≥50% violation rate
            'ford_roman_factor': 1e3,   # ≥10³ Ford-Roman violation
            'qi_efficiency': 0.1        # ≥10% quantum interest efficiency
        }
        
        # Check achievements
        achievements = {
            'anec_target_met': abs(anec_analysis.get('corrected_anec', 0)) >= targets['anec_magnitude'],
            'violation_rate_met': tree_results.get('mean_violation_rate', 0) >= targets['violation_rate'],
            'ford_roman_met': tree_results.get('max_ford_roman', 0) >= targets['ford_roman_factor'],
            'qi_efficiency_met': qi_results.get('best_efficiency', 0) >= targets['qi_efficiency'],
            'anec_negative': anec_analysis.get('corrected_anec', 0) < 0,
            'corrections_stable': anec_analysis.get('anec_sign_preserved', False)
        }
        
        # Overall assessment
        theory_targets_met = all([
            achievements['anec_target_met'],
            achievements['anec_negative'],
            achievements['violation_rate_met'],
            achievements['ford_roman_met']
        ])
        
        ready_for_hardware = theory_targets_met and achievements['corrections_stable']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'targets': targets,
            'achievements': achievements,
            'theory_targets_met': theory_targets_met,
            'ready_for_hardware': ready_for_hardware,
            'tree_level_results': tree_results,
            'radiative_corrections': corrections,
            'anec_analysis': anec_analysis,
            'quantum_interest': qi_results,
            'recommendation': self.generate_recommendation(theory_targets_met, ready_for_hardware, achievements)
        }
        
        self.results['summary'] = summary
        
        # Log summary
        logger.info("="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"ANEC Target (≥10⁵): {'✓' if achievements['anec_target_met'] else '✗'} "
                   f"({abs(anec_analysis.get('corrected_anec', 0)):.2e})")
        logger.info(f"Negative ANEC: {'✓' if achievements['anec_negative'] else '✗'}")
        logger.info(f"Violation Rate (≥50%): {'✓' if achievements['violation_rate_met'] else '✗'} "
                   f"({tree_results.get('mean_violation_rate', 0)*100:.1f}%)")
        logger.info(f"Ford-Roman (≥10³): {'✓' if achievements['ford_roman_met'] else '✗'} "
                   f"({tree_results.get('max_ford_roman', 0):.1e})")
        logger.info(f"QI Efficiency: {qi_results.get('best_efficiency', 0)*100:.1f}%")
        logger.info(f"Theory Targets Met: {'✓' if theory_targets_met else '✗'}")
        logger.info(f"Ready for Hardware: {'✓' if ready_for_hardware else '✗'}")
        logger.info("="*60)
        
        return summary
    
    def generate_recommendation(self, theory_met: bool, hardware_ready: bool, 
                              achievements: Dict) -> str:
        """Generate recommendations based on validation results."""
        if hardware_ready:
            return ("🎯 THEORY VALIDATED! All targets met with robust radiative stability. "
                   "Proceed to hardware prototyping and vacuum engineering.")
        elif theory_met:
            if not achievements['corrections_stable']:
                return ("⚠️ Theory targets met but radiative corrections destabilize ANEC sign. "
                       "Refine loop calculations and polymer prescriptions before hardware.")
            else:
                return ("🔬 Theory targets met. Optimize quantum-interest efficiency before hardware.")
        else:
            missing = []
            if not achievements['anec_target_met']:
                missing.append("ANEC magnitude")
            if not achievements['violation_rate_met']:
                missing.append("violation rate")
            if not achievements['ford_roman_met']:
                missing.append("Ford-Roman factor")
            
            return (f"❌ Theory targets not met. Focus on improving: {', '.join(missing)}. "
                   "Expand parameter space, refine polymer algebra, or explore alternative ansatz.")
    
    def save_results(self, summary: Dict):
        """Save all results to files."""
        # Save complete results as JSON
        results_file = self.output_dir / f"complete_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(self.results), f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save summary report
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("NEGATIVE ENERGY GENERATION - VALIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {summary['timestamp']}\n\n")
            
            f.write("THEORY TARGETS:\n")
            for target, value in summary['targets'].items():
                f.write(f"  {target}: {value}\n")
            f.write("\n")
            
            f.write("ACHIEVEMENTS:\n")
            for achievement, met in summary['achievements'].items():
                f.write(f"  {achievement}: {'✓' if met else '✗'}\n")
            f.write("\n")
            
            f.write("RECOMMENDATION:\n")
            f.write(f"  {summary['recommendation']}\n\n")
            
            if summary['tree_level_results'].get('best_params'):
                params = summary['tree_level_results']['best_params']
                f.write("OPTIMAL PARAMETERS:\n")
                f.write(f"  μ = {params['mu']:.6f}\n")
                f.write(f"  R = {params['R']:.6f}\n") 
                f.write(f"  τ = {params['tau']:.6f}\n")
                f.write(f"  ANEC = {summary['anec_analysis'].get('corrected_anec', 0):.2e} J·s·m⁻³\n")
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Run the complete validation pipeline."""
    print("🚀 NEGATIVE ENERGY GENERATION - COMPLETE VALIDATION PIPELINE")
    print("="*80)
    print("Integrating: Tree-level + Radiative corrections + Quantum interest")
    print("Targeting: ANEC < -10⁵ J·s·m⁻³, >50% violation rate, >10³× Ford-Roman")
    print("="*80)
    
    # Initialize and run pipeline
    pipeline = CompleteValidationPipeline()
    results = pipeline.run_complete_validation()
    
    # Display final results
    print("\n" + "="*80)
    print("🎯 FINAL VALIDATION RESULTS")
    print("="*80)
    
    if results['ready_for_hardware']:
        print("✅ SUCCESS: All theory targets met! Ready for hardware prototyping.")
    elif results['theory_targets_met']:
        print("⚡ PROGRESS: Theory targets met, refinements needed before hardware.")
    else:
        print("🔬 CONTINUE: Theory development required, targets not yet achieved.")
    
    print(f"\n📊 Recommendation: {results['recommendation']}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
