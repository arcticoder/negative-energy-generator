#!/usr/bin/env python3
"""
Parallel Development Controller
==============================

Coordinates theory refinement and experimental prototyping in parallel.
Enforces strict go/no-go criteria for full demonstrator transition.

Only proceeds to full demonstrator when BOTH:
- best_anec_2d ≤ -1e5 J·s·m⁻³  
- best_rate_2d ≥ 0.5 (50%)

Until then: PARALLEL_DEVELOPMENT mode

Usage:
    python parallel_development_controller.py
"""

import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src', 'prototype'))

try:
    from iterative_theory_refinement import check_scan_results, assess_readiness, theory_iteration_loop
    from src.prototype.combined_prototype import combined_prototype_demonstration, UnifiedVacuumGenerator
    from src.prototype.integrated_derisking_suite import integrated_derisking_assessment
except ImportError:
    # Fallback for demo purposes
    print("⚠️ Import paths not found - using mock functions for demonstration")
    
    def check_scan_results():
        return {'best_anec_2d': -2.09e-6, 'best_rate_2d': 0.42, 'scan_iteration': 1}
    
    def assess_readiness(results):
        return {
            'anec_met': False, 'rate_met': False, 
            'anec_gap': 47846889952, 'rate_gap': 0.08,
            'best_anec_2d': results['best_anec_2d'],
            'best_rate_2d': results['best_rate_2d']
        }
    
    def theory_iteration_loop(max_iter):
        return {'status': 'PARALLEL_DEVELOPMENT', 'iterations': 3}
    
    def combined_prototype_demonstration():
        return {
            'readiness': {
                'casimir': {'ready': True, 'readiness_pct': 85},
                'dynamic': {'ready': False, 'readiness_pct': 60}, 
                'squeezed': {'ready': False, 'readiness_pct': 45},
                'metamaterial': {'ready': True, 'readiness_pct': 75},
                'overall': {'ready_components': 2, 'total_components': 4, 'overall_ready': False, 'readiness_pct': 50}
            }
        }
    
    class UnifiedVacuumGenerator:
        def assess_prototype_readiness(self):
            return combined_prototype_demonstration()['readiness']
        def run_parallel_optimization_scans(self):
            return {'casimir': {}, 'dynamic': {}, 'squeezed': {}, 'metamaterial': {}}
    
    def integrated_derisking_assessment():
        return {
            'overall_assessment': {
                'risk_level': 'LOW-MEDIUM',
                'recommendation': 'Proceed with prototyping'
            }
        }

def check_full_demonstrator_readiness():
    """Check if ready for full demonstrator based on strict criteria."""
    
    print("🚦 FULL DEMONSTRATOR READINESS CHECK")
    print("=" * 36)
    print()
    
    # Check theory readiness
    results = check_scan_results()
    status = assess_readiness(results)
    
    print("📊 THEORY STATUS:")
    print(f"  ANEC: {results['best_anec_2d']:.2e} J·s·m⁻³")
    print(f"  Rate: {results['best_rate_2d']*100:.1f}%")
    print()
    
    anec_status = "✅ MET" if status['anec_met'] else "❌ NOT MET"
    rate_status = "✅ MET" if status['rate_met'] else "❌ NOT MET"
    
    print("🎯 CRITERIA EVALUATION:")
    print(f"  ANEC ≤ -1e5: {anec_status}")
    print(f"  Rate ≥ 50%: {rate_status}")
    print()
    
    # Check experimental readiness
    generator = UnifiedVacuumGenerator()
    readiness = generator.assess_prototype_readiness()
    
    print("🔬 EXPERIMENTAL STATUS:")
    experimental_ready = readiness['overall']['overall_ready']
    exp_status = "✅ READY" if experimental_ready else "🔄 DEVELOPING"
    print(f"  Prototypes: {exp_status} ({readiness['overall']['ready_components']}/4 ready)")
    print()
    
    # Overall decision
    theory_ready = status['anec_met'] and status['rate_met']
    full_ready = theory_ready and experimental_ready
    
    print("🚦 FINAL DECISION:")
    if full_ready:
        decision = "🟢 READY FOR FULL DEMONSTRATOR"
        mode = "FULL_DEMONSTRATOR"
        print(f"   {decision}")
        print("   ✅ Both theory and experimental criteria satisfied")
        print("   🚀 Proceed with large-scale integration")
    else:
        decision = "🟡 CONTINUE PARALLEL DEVELOPMENT"
        mode = "PARALLEL_DEVELOPMENT"
        print(f"   {decision}")
        if not theory_ready:
            print("   ⚠️ Theory targets not yet achieved")
            if not status['anec_met']:
                print(f"      • ANEC gap: {status['anec_gap']:.1e}× improvement needed")
            if not status['rate_met']:
                print(f"      • Rate gap: {status['rate_gap']*100:.1f} percentage points")
        if not experimental_ready:
            print("   ⚠️ Experimental modules need further development")
    
    print()
    
    return {
        'mode': mode,
        'theory_ready': theory_ready,
        'experimental_ready': experimental_ready,
        'full_ready': full_ready,
        'theory_status': status,
        'experimental_status': readiness
    }

def run_parallel_development_cycle():
    """Run one cycle of parallel theory-experiment development."""
    
    print("🔄 PARALLEL DEVELOPMENT CYCLE")
    print("=" * 30)
    print()
    
    results = {}
    
    # 1. Theory refinement iteration
    print("🧮 THEORY TRACK")
    print("-" * 13)
    from iterative_theory_refinement import refine_theoretical_model, update_theory_results
    
    current_results = check_scan_results()
    refinement = refine_theoretical_model(current_results.get('scan_iteration', 1))
    updated_results = update_theory_results(current_results, refinement)
    
    results['theory'] = {
        'previous': current_results,
        'refinement': refinement,
        'updated': updated_results
    }
    print()
    
    # 2. Experimental optimization
    print("🔬 EXPERIMENT TRACK")
    print("-" * 16)
    
    generator = UnifiedVacuumGenerator()
    scan_results = generator.run_parallel_optimization_scans()
    readiness = generator.assess_prototype_readiness()
    
    results['experiments'] = {
        'scan_results': scan_results,
        'readiness': readiness
    }
    
    # 3. Risk assessment
    print("🛡️ VALIDATION TRACK")
    print("-" * 17)
    
    risk_assessment = integrated_derisking_assessment()
    
    results['validation'] = risk_assessment
    print()
    
    return results

def parallel_development_loop(max_cycles=10):
    """Main parallel development loop."""
    
    print("🎯 PARALLEL DEVELOPMENT COORDINATION")
    print("=" * 37)
    print()
    print("Coordinating theory refinement + experimental prototyping")
    print("until both ANEC and violation rate targets are achieved.")
    print()
    
    cycle = 1
    
    while cycle <= max_cycles:
        print(f"🔄 DEVELOPMENT CYCLE {cycle}")
        print("=" * 20)
        
        # Check current readiness
        readiness_check = check_full_demonstrator_readiness()
        
        if readiness_check['full_ready']:
            print("🎉 BREAKTHROUGH ACHIEVED!")
            print("Both theory and experimental targets met!")
            print("🚀 Proceeding to full demonstrator construction...")
            return {
                'status': 'READY',
                'cycles_completed': cycle,
                'final_readiness': readiness_check
            }
        
        print(f"Continuing parallel development (cycle {cycle})...")
        print()
        
        # Run development cycle
        cycle_results = run_parallel_development_cycle()
        
        # Summary of cycle progress
        theory_progress = cycle_results['theory']['refinement']
        exp_progress = cycle_results['experiments']['readiness']['overall']
        
        print("📈 CYCLE PROGRESS SUMMARY:")
        print(f"  Theory: {theory_progress['strategy']}")
        print(f"  Expected ANEC improvement: {theory_progress['anec_improvement']:.2f}×")
        print(f"  Expected rate improvement: +{theory_progress['rate_improvement']*100:.1f}%")
        print(f"  Experimental readiness: {exp_progress['readiness_pct']:.0f}%")
        print()
        
        cycle += 1
    
    # Max cycles reached
    print("⏰ Maximum development cycles reached")
    print("🟡 Continuing in PARALLEL_DEVELOPMENT mode")
    
    final_check = check_full_demonstrator_readiness()
    return {
        'status': 'PARALLEL_DEVELOPMENT',
        'cycles_completed': max_cycles,
        'final_readiness': final_check
    }

def show_development_status():
    """Show comprehensive development status across all tracks."""
    
    print("📊 COMPREHENSIVE DEVELOPMENT STATUS")
    print("=" * 35)
    print()
    
    # Theory status
    results = check_scan_results()
    status = assess_readiness(results)
    
    print("🧮 THEORY TRACK STATUS:")
    print(f"  Current ANEC: {results['best_anec_2d']:.2e} J·s·m⁻³")
    print(f"  Current rate: {results['best_rate_2d']*100:.1f}%")
    print(f"  ANEC gap: {status['anec_gap']:.0f}× improvement needed")
    print(f"  Rate gap: {status['rate_gap']*100:.1f} percentage points") 
    print(f"  Theory ready: {'✅ YES' if status['anec_met'] and status['rate_met'] else '❌ NO'}")
    print()
    
    # Experimental status
    demo_results = combined_prototype_demonstration()
    readiness = demo_results['readiness']
    
    print("🔬 EXPERIMENTAL TRACK STATUS:")
    for component, assessment in readiness.items():
        if component == 'overall':
            continue
        status_icon = "✅" if assessment['ready'] else "⚠️"
        print(f"  {status_icon} {component.capitalize()}: {assessment['readiness_pct']:.0f}% ready")
    
    overall = readiness['overall']
    print(f"  Overall: {overall['ready_components']}/{overall['total_components']} modules ready")
    print()
    
    # Risk status
    print("🛡️ VALIDATION TRACK STATUS:")
    risk_assessment = integrated_derisking_assessment()
    overall_risk = risk_assessment['overall_assessment']['risk_level']
    recommendation = risk_assessment['overall_assessment']['recommendation']
    
    print(f"  Overall risk: {overall_risk}")
    print(f"  Recommendation: {recommendation}")
    print()
    
    # Coordination status
    print("🎯 COORDINATION STATUS:")
    readiness_check = check_full_demonstrator_readiness()
    print(f"  Development mode: {readiness_check['mode']}")
    
    if readiness_check['mode'] == 'PARALLEL_DEVELOPMENT':
        print("  🔄 Continue theory + experiment in parallel")
        print("  🎯 Work toward both ANEC ≤ -1e5 AND rate ≥ 50%")
    else:
        print("  🚀 Ready for full demonstrator!")
    
    print()

def main():
    """Main parallel development controller."""
    
    print("🎛️ PARALLEL DEVELOPMENT CONTROLLER")
    print("=" * 35)
    print()
    
    # Show current comprehensive status
    show_development_status()
    
    # Check readiness for full demonstrator
    readiness_check = check_full_demonstrator_readiness()
    
    if readiness_check['full_ready']:
        print("🎉 READY FOR FULL DEMONSTRATOR!")
        print("Both theory and experimental criteria satisfied.")
        return readiness_check
    
    # Run development cycles
    print("🔄 Starting parallel development cycles...")
    print("(Limited to 3 cycles for demonstration)")
    print()
    
    result = parallel_development_loop(max_cycles=3)
    
    print("=" * 35)
    print("🏁 DEVELOPMENT SUMMARY")
    print("=" * 35)
    print(f"Status: {result['status']}")
    print(f"Cycles completed: {result['cycles_completed']}")
    
    if result['status'] == 'READY':
        print("🚀 Ready for full demonstrator construction!")
    else:
        print("🔄 Continue parallel development")
        print("🎯 Focus: Close theory gaps while optimizing prototypes")
    
    return result

if __name__ == "__main__":
    main()
