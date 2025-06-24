#!/usr/bin/env python3
"""
Phase 2 Readiness Assessment with Go/No-Go Criteria
===================================================

This script implements the strict go/no-go criteria for full demonstrator approval:

CRITERIA FOR "READY":
- anec_met = best_anec_2d <= -1e5 (ANEC magnitude more negative than −10⁵ J·s·m⁻³)
- rate_met = best_rate_2d >= 0.50 (≥50% violation rate)

Only when BOTH criteria are met does the system return "READY".
Otherwise: "PARALLEL_DEVELOPMENT" (continue theory AND build testbeds).

Usage:
    python phase2_readiness_check.py
"""

import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def load_scan_results():
    """Load the latest ANEC and violation rate scan results."""
    
    # Try to load actual scan data
    scan_files = [
        "../../lqg-anec-framework/2d_parameter_sweep_complete.json",
        "../../lqg-anec-framework/2d_parameter_sweep_summary.csv",
        "advanced_scan_results.json",
        "scan_results.json"
    ]
    
    best_anec_2d = None
    best_rate_2d = None
    
    # Try to load from various sources
    for scan_file in scan_files:
        full_path = os.path.join(os.path.dirname(__file__), scan_file)
        if os.path.exists(full_path):
            try:
                if scan_file.endswith('.json'):
                    import json
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract best values from different possible formats
                    if 'best_anec_magnitude' in data:
                        best_anec_2d = data['best_anec_magnitude']
                    elif 'results' in data and len(data['results']) > 0:
                        results = data['results']
                        anec_values = [r.get('anec_magnitude', 0) for r in results]
                        best_anec_2d = min(anec_values) if anec_values else None
                    
                    if 'best_violation_rate' in data:
                        best_rate_2d = data['best_violation_rate']
                    elif 'results' in data and len(data['results']) > 0:
                        results = data['results']
                        rate_values = [r.get('violation_rate', 0) for r in results]
                        best_rate_2d = max(rate_values) if rate_values else None
                
                elif scan_file.endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(full_path)
                        
                        # Look for ANEC and rate columns
                        anec_columns = [c for c in df.columns if 'anec' in c.lower()]
                        rate_columns = [c for c in df.columns if 'rate' in c.lower() or 'violation' in c.lower()]
                        
                        if anec_columns:
                            best_anec_2d = df[anec_columns[0]].min()
                        if rate_columns:
                            best_rate_2d = df[rate_columns[0]].max()
                    except ImportError:
                        # Fallback without pandas
                        pass
                
                if best_anec_2d is not None and best_rate_2d is not None:
                    print(f"✅ Loaded scan data from {scan_file}")
                    break
                    
            except Exception as e:
                print(f"⚠️ Could not load {scan_file}: {e}")
                continue
    
    # If no data found, use current best estimates from previous assessments
    if best_anec_2d is None or best_rate_2d is None:
        print("📊 Using current best estimates (no scan data found)")
        best_anec_2d = -2.09e-6  # Current best from previous assessment
        best_rate_2d = 0.42      # Current best violation rate (42%)
    
    return best_anec_2d, best_rate_2d

def evaluate_readiness_criteria(best_anec_2d: float, best_rate_2d: float) -> dict:
    """
    Evaluate go/no-go criteria for full demonstrator.
    
    Args:
        best_anec_2d: Best ANEC magnitude [J·s·m⁻³]
        best_rate_2d: Best violation rate [0-1]
        
    Returns:
        Readiness assessment dictionary
    """
    
    # Define target criteria
    ANEC_TARGET = -1e5      # −10⁵ J·s·m⁻³
    RATE_TARGET = 0.50      # 50% violation rate
    
    # Evaluate criteria
    anec_met = best_anec_2d <= ANEC_TARGET
    rate_met = best_rate_2d >= RATE_TARGET
    
    # Overall readiness
    ready_for_full_demonstrator = anec_met and rate_met
    
    # Calculate gaps
    anec_gap_factor = abs(ANEC_TARGET / best_anec_2d) if best_anec_2d != 0 else float('inf')
    rate_gap_points = RATE_TARGET - best_rate_2d
    
    return {
        'best_anec_2d': best_anec_2d,
        'best_rate_2d': best_rate_2d,
        'anec_target': ANEC_TARGET,
        'rate_target': RATE_TARGET,
        'anec_met': anec_met,
        'rate_met': rate_met,
        'ready_for_full_demonstrator': ready_for_full_demonstrator,
        'anec_gap_factor': anec_gap_factor,
        'rate_gap_points': rate_gap_points,
        'development_mode': 'READY' if ready_for_full_demonstrator else 'PARALLEL_DEVELOPMENT'
    }

def generate_readiness_report(assessment: dict) -> str:
    """Generate comprehensive readiness report."""
    
    report = []
    report.append("🎯 PHASE 2 READINESS ASSESSMENT")
    report.append("=" * 34)
    report.append("")
    
    # Current status
    report.append("📊 CURRENT THEORETICAL STATUS")
    report.append("-" * 29)
    report.append(f"Best ANEC magnitude: {assessment['best_anec_2d']:.2e} J·s·m⁻³")
    report.append(f"Best violation rate: {assessment['best_rate_2d']*100:.1f}%")
    report.append("")
    
    # Target criteria
    report.append("🎯 TARGET CRITERIA")
    report.append("-" * 17)
    report.append(f"ANEC target: ≤ {assessment['anec_target']:.0e} J·s·m⁻³")
    report.append(f"Rate target: ≥ {assessment['rate_target']*100:.0f}%")
    report.append("")
    
    # Criteria assessment
    report.append("✅ CRITERIA ASSESSMENT")
    report.append("-" * 21)
    
    anec_status = "✅ MET" if assessment['anec_met'] else "❌ NOT MET"
    rate_status = "✅ MET" if assessment['rate_met'] else "❌ NOT MET"
    
    report.append(f"ANEC criterion: {anec_status}")
    if not assessment['anec_met']:
        report.append(f"  Gap: {assessment['anec_gap_factor']:.0f}× improvement needed")
    
    report.append(f"Rate criterion: {rate_status}")
    if not assessment['rate_met']:
        report.append(f"  Gap: {assessment['rate_gap_points']*100:.1f} percentage points")
    
    report.append("")
    
    # Final decision
    report.append("🚦 FINAL DECISION")
    report.append("-" * 15)
    
    if assessment['ready_for_full_demonstrator']:
        report.append("🟢 STATUS: READY")
        report.append("")
        report.append("✅ BOTH criteria met - APPROVED for full demonstrator")
        report.append("")
        report.append("🚀 NEXT ACTIONS:")
        report.append("   1. Begin unified large-scale demonstrator construction")
        report.append("   2. Integrate all vacuum-engineering modules")
        report.append("   3. Scale to engineering-relevant energy levels")
        report.append("   4. Prepare for practical applications")
        report.append("   5. Document reproducible protocols")
    else:
        report.append("🟡 STATUS: PARALLEL_DEVELOPMENT")
        report.append("")
        report.append("⚠️ Criteria not yet met - CONTINUE parallel development")
        report.append("")
        report.append("📋 PARALLEL TRACKS:")
        report.append("   🧮 THEORY: Continue ANEC violation optimization")
        if not assessment['anec_met']:
            report.append(f"      → Target: {assessment['anec_gap_factor']:.0f}× improvement in magnitude")
        if not assessment['rate_met']:
            report.append(f"      → Target: +{assessment['rate_gap_points']*100:.1f} percentage points in rate")
        report.append("")
        report.append("   🔬 EXPERIMENT: Build and validate testbeds")
        report.append("      → Casimir array demonstrators")
        report.append("      → Dynamic Casimir cavities")
        report.append("      → Squeezed vacuum sources")
        report.append("      → Metamaterial enhancement")
        report.append("")
        report.append("   🛡️ VALIDATION: De-risk experimental approach")
        report.append("      → Uncertainty quantification")
        report.append("      → Bayesian optimization")
        report.append("      → Real-time monitoring")
        report.append("      → Sensitivity analysis")
    
    report.append("")
    report.append("=" * 34)
    report.append(f"Development Mode: {assessment['development_mode']}")
    report.append("=" * 34)
    
    return "\n".join(report)

def assess_prototype_readiness():
    """Assess readiness of Phase 2 prototypes independent of theory."""
    
    try:
        from prototype.integrated_derisking_suite import comprehensive_derisking_analysis, generate_integration_summary
        
        print("🛡️ PROTOTYPE READINESS ASSESSMENT")
        print("=" * 34)
        print()
        
        # Run de-risking analysis
        try:
            results = comprehensive_derisking_analysis()
            summary = generate_integration_summary(results)
            
            prototype_ready = summary['ready_for_prototype']
            overall_risk = summary['overall_risk']
            
            print(f"🔬 Prototype Risk Level: {overall_risk}")
            print(f"✅ Prototype Ready: {prototype_ready}")
            print()
            
            if prototype_ready:
                print("✅ PROTOTYPES: Ready for experimental construction")
                print("   → All risk factors well-characterized")
                print("   → De-risking framework operational")
                print("   → Validation tools ready")
            else:
                print("⚠️ PROTOTYPES: Additional de-risking needed")
                print("   → Address remaining risk factors")
                print("   → Improve measurement capabilities")
                print("   → Refine design parameters")
            
            return {
                'prototype_ready': prototype_ready,
                'overall_risk': overall_risk,
                'risk_summary': summary
            }
            
        except Exception as e:
            print(f"❌ De-risking analysis failed: {e}")
            return {
                'prototype_ready': False,
                'overall_risk': 'UNKNOWN',
                'risk_summary': None
            }
            
    except ImportError as e:
        print(f"⚠️ De-risking suite unavailable: {e}")
        return {
            'prototype_ready': False,
            'overall_risk': 'UNKNOWN',
            'risk_summary': None
        }

def main():
    """Main readiness assessment with go/no-go criteria."""
    
    print("🎯 NEGATIVE ENERGY GENERATOR: PHASE 2 READINESS CHECK")
    print("=" * 54)
    print()
    print("Implementing strict go/no-go criteria for full demonstrator...")
    print()
    
    # Load scan results
    print("📊 Loading theoretical assessment data...")
    best_anec_2d, best_rate_2d = load_scan_results()
    print()
    
    # Evaluate readiness
    print("⚖️ Evaluating readiness criteria...")
    assessment = evaluate_readiness_criteria(best_anec_2d, best_rate_2d)
    print()
    
    # Generate and display report
    report = generate_readiness_report(assessment)
    print(report)
    print()
    
    # Assess prototype readiness separately
    prototype_assessment = assess_prototype_readiness()
    print()
    
    # Combined recommendation
    print("🎯 COMBINED RECOMMENDATION")
    print("=" * 25)
    print()
    
    theory_ready = assessment['ready_for_full_demonstrator']
    prototype_ready = prototype_assessment['prototype_ready']
    
    if theory_ready and prototype_ready:
        print("🟢 FULL GO: Theory + Prototypes ready")
        print("   → Proceed with unified large-scale demonstrator")
    elif theory_ready and not prototype_ready:
        print("🟡 THEORY GO, PROTOTYPE CAUTION:")
        print("   → Theory targets met, but prototype de-risking needed")
        print("   → Complete prototype validation, then proceed")
    elif not theory_ready and prototype_ready:
        print("🟡 PROTOTYPE GO, THEORY CONTINUE:")
        print("   → Prototypes ready, but theory targets not met")
        print("   → Build testbeds while continuing theory optimization")
    else:
        print("🟡 PARALLEL DEVELOPMENT:")
        print("   → Continue theory AND prototype development")
        print("   → Neither track ready for full demonstrator")
    
    print()
    print("=" * 54)
    print(f"🚦 FINAL STATUS: {assessment['development_mode']}")
    print("=" * 54)
    
    return assessment

if __name__ == "__main__":
    main()
