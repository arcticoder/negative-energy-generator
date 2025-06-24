#!/usr/bin/env python3
"""
Integrated Vacuum Engineering De-Risking Suite
==============================================

This script demonstrates all four de-risking tools for vacuum-engineering prototypes:

1. 🔬 Uncertainty Quantification via Monte Carlo Propagation
2. 🤖 Bayesian Optimization of Design Parameters  
3. 📐 Sensitivity Analysis & Analytical Error Bounds
4. 📡 Real-Time Data Ingestion & Residual Analysis

Each tool helps reduce experimental risk while theory continues to improve.

Usage:
    python integrated_derisking_suite.py
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from prototype.error_analysis import uncertainty_report
    from prototype.bayesian_optimization import optimization_comparison
    from prototype.sensitivity import dimensional_analysis_report
    from prototype.data_residuals import real_time_monitoring_report, create_mock_data, compute_predicted
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Ensure all prototype modules are in src/prototype/")

def comprehensive_derisking_analysis():
    """Run comprehensive de-risking analysis for Casimir array prototype."""
    
    print("🛡️ INTEGRATED VACUUM ENGINEERING DE-RISKING SUITE")
    print("=" * 52)
    print()
    
    # Standard test configuration
    test_gaps = np.array([6e-9, 7e-9, 8e-9, 7e-9, 6e-9])  # 5 gaps, slight variation
    gap_uncertainty = 1e-10  # ±0.1 nm precision
    
    print(f"🔧 TEST CONFIGURATION")
    print(f"-" * 18)
    print(f"Gap configuration: {test_gaps * 1e9} nm")
    print(f"Gap uncertainty: ±{gap_uncertainty * 1e9:.1f} nm")
    print(f"Number of gaps: {len(test_gaps)}")
    print()
    
    # 1. Uncertainty Quantification
    print("=" * 52)
    print("1️⃣ UNCERTAINTY QUANTIFICATION")
    print("=" * 52)
    
    try:
        uncertainty_results = uncertainty_report(test_gaps, gap_uncertainty)
        print("✅ Uncertainty analysis completed")
    except Exception as e:
        print(f"❌ Uncertainty analysis failed: {e}")
        uncertainty_results = None
    
    print()
    
    # 2. Bayesian Optimization
    print("=" * 52)
    print("2️⃣ BAYESIAN OPTIMIZATION")
    print("=" * 52)
    
    try:
        optimization_results = optimization_comparison(
            N=5, 
            d_min=5e-9, 
            d_max=10e-9
        )
        print("✅ Bayesian optimization completed")
    except Exception as e:
        print(f"❌ Bayesian optimization failed: {e}")
        optimization_results = None
    
    print()
    
    # 3. Sensitivity Analysis
    print("=" * 52)
    print("3️⃣ SENSITIVITY ANALYSIS")
    print("=" * 52)
    
    try:
        sensitivity_results = dimensional_analysis_report(test_gaps, sigma_rel=0.02)
        print("✅ Sensitivity analysis completed")
    except Exception as e:
        print(f"❌ Sensitivity analysis failed: {e}")
        sensitivity_results = None
    
    print()
    
    # 4. Real-Time Data Analysis
    print("=" * 52)
    print("4️⃣ REAL-TIME DATA MONITORING")
    print("=" * 52)
    
    try:
        # Create mock experimental data
        mock_data = create_mock_data(n_samples=100)
        mock_data = compute_predicted(mock_data)
        
        monitoring_results = real_time_monitoring_report(mock_data)
        print("✅ Data monitoring analysis completed")
    except Exception as e:
        print(f"❌ Data monitoring failed: {e}")
        monitoring_results = None
    
    print()
    
    return {
        'uncertainty': uncertainty_results,
        'optimization': optimization_results,
        'sensitivity': sensitivity_results,
        'monitoring': monitoring_results
    }

def generate_integration_summary(results):
    """Generate integrated summary of all de-risking analyses."""
    
    print("=" * 52)
    print("📋 INTEGRATED DE-RISKING SUMMARY")
    print("=" * 52)
    print()
    
    # Risk assessment matrix
    risk_levels = {}
    
    # Uncertainty risk
    if results['uncertainty']:
        unc_risk = "LOW" if results['uncertainty']['mc_rel_error'] < 5 else "MEDIUM" if results['uncertainty']['mc_rel_error'] < 10 else "HIGH"
        risk_levels['uncertainty'] = unc_risk
        print(f"🔬 Uncertainty Risk: {unc_risk}")
        print(f"   Relative error: {results['uncertainty']['mc_rel_error']:.1f}%")
    else:
        risk_levels['uncertainty'] = "UNKNOWN"
        print(f"🔬 Uncertainty Risk: UNKNOWN (analysis failed)")
    
    # Optimization risk
    if results['optimization']:
        improvements = results['optimization'].get('improvements', {})
        max_improvement = max(improvements.values()) if improvements else 1.0
        opt_risk = "LOW" if max_improvement > 1.5 else "MEDIUM" if max_improvement > 1.1 else "HIGH"
        risk_levels['optimization'] = opt_risk
        print(f"🤖 Optimization Risk: {opt_risk}")
        print(f"   Best improvement: {max_improvement:.2f}×")
    else:
        risk_levels['optimization'] = "UNKNOWN"
        print(f"🤖 Optimization Risk: UNKNOWN (analysis failed)")
    
    # Sensitivity risk
    if results['sensitivity']:
        sens_risk = "LOW" if results['sensitivity']['condition_number'] < 1e3 else "MEDIUM" if results['sensitivity']['condition_number'] < 1e6 else "HIGH"
        risk_levels['sensitivity'] = sens_risk
        print(f"📐 Sensitivity Risk: {sens_risk}")
        print(f"   Condition number: {results['sensitivity']['condition_number']:.1e}")
    else:
        risk_levels['sensitivity'] = "UNKNOWN"
        print(f"📐 Sensitivity Risk: UNKNOWN (analysis failed)")
    
    # Monitoring risk
    if results['monitoring']:
        mon_risk = "LOW" if results['monitoring']['r_squared'] > 0.95 else "MEDIUM" if results['monitoring']['r_squared'] > 0.90 else "HIGH"
        risk_levels['monitoring'] = mon_risk
        print(f"📡 Monitoring Risk: {mon_risk}")
        print(f"   Model R²: {results['monitoring']['r_squared']:.3f}")
    else:
        risk_levels['monitoring'] = "UNKNOWN"
        print(f"📡 Monitoring Risk: UNKNOWN (analysis failed)")
    
    print()
    
    # Overall risk assessment
    print("🎯 OVERALL RISK ASSESSMENT")
    print("-" * 26)
    
    high_risks = sum(1 for risk in risk_levels.values() if risk == "HIGH")
    medium_risks = sum(1 for risk in risk_levels.values() if risk == "MEDIUM")
    low_risks = sum(1 for risk in risk_levels.values() if risk == "LOW")
    unknown_risks = sum(1 for risk in risk_levels.values() if risk == "UNKNOWN")
    
    if high_risks == 0 and unknown_risks == 0:
        overall_risk = "LOW"
        risk_color = "✅"
    elif high_risks <= 1 and unknown_risks == 0:
        overall_risk = "MEDIUM" 
        risk_color = "⚠️"
    else:
        overall_risk = "HIGH"
        risk_color = "❌"
    
    print(f"{risk_color} Overall Risk Level: {overall_risk}")
    print(f"   High risks: {high_risks}")
    print(f"   Medium risks: {medium_risks}")
    print(f"   Low risks: {low_risks}")
    print(f"   Unknown risks: {unknown_risks}")
    print()
    
    # Readiness recommendations
    print("💡 PROTOTYPE READINESS RECOMMENDATIONS")
    print("-" * 38)
    
    if overall_risk == "LOW":
        print("🚀 READY FOR PROTOTYPE CONSTRUCTION")
        print("   • All risk factors well-characterized")
        print("   • Proceed with fabrication planning")
        print("   • Implement real-time monitoring")
        print("   • Use optimized design parameters")
    elif overall_risk == "MEDIUM":
        print("⚠️ PROCEED WITH CAUTION")
        print("   • Address medium/high risk factors")
        print("   • Implement enhanced monitoring")
        print("   • Consider design modifications")
        print("   • Plan additional characterization")
    else:
        print("🛑 ADDITIONAL DE-RISKING REQUIRED")
        print("   • Resolve high-risk factors before prototyping")
        print("   • Improve measurement capabilities")
        print("   • Refine theoretical models")
        print("   • Conduct preliminary validation studies")
    
    print()
    
    # Next steps
    print("📋 RECOMMENDED NEXT STEPS")
    print("-" * 25)
    
    if results['uncertainty'] and results['uncertainty']['mc_rel_error'] > 5:
        print("🔬 Improve gap measurement precision")
        print(f"   Current: ±{results['uncertainty']['mc_rel_error']:.1f}%")
        print("   Target: <5% relative error")
    
    if results['optimization'] and max(results['optimization'].get('improvements', {1:1}).values()) < 1.2:
        print("🤖 Explore alternative gap configurations")
        print("   Current improvement modest")
        print("   Consider non-uniform designs")
    
    if results['sensitivity'] and results['sensitivity']['condition_number'] > 1e6:
        print("📐 Redesign for better numerical stability")
        print("   High condition number detected")
        print("   Consider more uniform gap distribution")
    
    if results['monitoring'] and results['monitoring']['r_squared'] < 0.95:
        print("📡 Improve theoretical model accuracy")
        print(f"   Current R²: {results['monitoring']['r_squared']:.3f}")
        print("   Target: >0.95 for reliable prediction")
    
    return {
        'overall_risk': overall_risk,
        'risk_levels': risk_levels,
        'ready_for_prototype': overall_risk in ["LOW", "MEDIUM"]
    }

def main():
    """Main integrated de-risking analysis."""
    
    # Run comprehensive analysis
    results = comprehensive_derisking_analysis()
    
    # Generate integrated summary
    summary = generate_integration_summary(results)
    
    # Final assessment
    print("=" * 52)
    print("🏁 FINAL ASSESSMENT")
    print("=" * 52)
    
    if summary['ready_for_prototype']:
        print("✅ PROTOTYPE CONSTRUCTION APPROVED")
        print()
        print("🔧 Immediate Actions:")
        print("   1. Finalize fabrication specifications")
        print("   2. Set up measurement infrastructure")
        print("   3. Implement real-time monitoring")
        print("   4. Begin prototype assembly")
        print()
        print("⚡ Parallel Activities:")
        print("   • Continue theoretical ANEC optimization")
        print("   • Validate other vacuum-engineering modules")
        print("   • Prepare integration protocols")
        print("   • Document experimental procedures")
    else:
        print("⚠️ ADDITIONAL PREPARATION REQUIRED")
        print()
        print("🔄 Priority Actions:")
        print("   1. Address high-risk factors identified above")
        print("   2. Improve measurement capabilities")
        print("   3. Refine design parameters")
        print("   4. Conduct validation studies")
        print()
        print("📚 Continue theoretical development while preparing")
    
    print()
    print("=" * 52)
    print("🛡️ DE-RISKING ANALYSIS COMPLETE")
    print("=" * 52)
    
    return summary

if __name__ == "__main__":
    main()
