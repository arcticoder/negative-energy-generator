# src/prototype/data_residuals.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

hbar, c = 1.054e-34, 3e8

def load_measurements(path):
    """Load experimental measurements from CSV."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {len(df)} measurements from {path}")
        return df
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return create_mock_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return create_mock_data()

def create_mock_data(n_samples=50):
    """Create mock experimental data for demonstration."""
    print("📊 Creating mock experimental data...")
    
    np.random.seed(42)  # Reproducible results
    
    # Generate realistic gap measurements with noise
    true_gaps = np.random.uniform(6e-9, 10e-9, size=(n_samples, 5))
    gap_noise = np.random.normal(0, 1e-10, size=(n_samples, 5))
    measured_gaps = true_gaps + gap_noise
    
    # Calculate true energy
    true_energy = np.array([
        np.sum(- (np.pi**2 * hbar * c)/(720 * gaps**3)) 
        for gaps in true_gaps
    ])
    
    # Add measurement noise to energy
    energy_noise = np.random.normal(0, abs(true_energy) * 0.05)  # 5% noise
    measured_energy = true_energy + energy_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'd1': measured_gaps[:, 0],
        'd2': measured_gaps[:, 1], 
        'd3': measured_gaps[:, 2],
        'd4': measured_gaps[:, 3],
        'd5': measured_gaps[:, 4],
        'E_meas': measured_energy,
        'timestamp': pd.date_range(start='2025-06-01', periods=n_samples, freq='H')
    })
    
    return df

def compute_predicted(df):
    """Compute predicted energies from gap measurements."""
    def E_row(row):
        ds = row[['d1','d2','d3','d4','d5']].values
        return np.sum(- (np.pi**2 * hbar * c)/(720 * ds**3))
    
    df['E_pred'] = df.apply(E_row, axis=1)
    return df

def residuals_analysis(df):
    """Comprehensive residual analysis."""
    df['residuals'] = df['E_meas'] - df['E_pred']
    df['rel_residuals'] = df['residuals'] / df['E_pred'] * 100
    
    stats = {
        'mean': df['residuals'].mean(),
        'std': df['residuals'].std(),
        'rms': np.sqrt(np.mean(df['residuals']**2)),
        'max_abs': np.max(np.abs(df['residuals'])),
        'rel_mean': df['rel_residuals'].mean(),
        'rel_std': df['rel_residuals'].std()
    }
    
    return stats

def drift_detection(df, window_size=10):
    """Detect systematic drifts in measurements."""
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Rolling statistics
    df['rolling_residual_mean'] = df['residuals'].rolling(window=window_size).mean()
    df['rolling_residual_std'] = df['residuals'].rolling(window=window_size).std()
    
    # Detect trend
    if len(df) > 20:
        from scipy import stats
        x = np.arange(len(df))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, df['residuals'])
        
        drift_info = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant_drift': p_value < 0.05 and abs(r_value) > 0.3
        }
    else:
        drift_info = {'significant_drift': False}
    
    return drift_info

def outlier_detection(df, threshold=3.0):
    """Detect outliers using z-score method."""
    z_scores = np.abs((df['residuals'] - df['residuals'].mean()) / df['residuals'].std())
    outliers = df[z_scores > threshold]
    
    return outliers

def real_time_monitoring_report(df):
    """Generate comprehensive real-time monitoring report."""
    
    print("📡 REAL-TIME DATA MONITORING REPORT")
    print("=" * 38)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Data overview
    print("📊 DATA OVERVIEW")
    print("-" * 16)
    print(f"Total measurements: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "No timestamps")
    print(f"Gap ranges:")
    for i in range(1, 6):
        col = f'd{i}'
        print(f"  Gap {i}: {df[col].min()*1e9:.1f} - {df[col].max()*1e9:.1f} nm")
    print()
    
    # Residual statistics
    print("📈 RESIDUAL ANALYSIS")
    print("-" * 19)
    
    stats = residuals_analysis(df)
    
    print(f"Residual mean: {stats['mean']:.3e} J/m²")
    print(f"Residual std: {stats['std']:.3e} J/m²") 
    print(f"RMS error: {stats['rms']:.3e} J/m²")
    print(f"Max absolute error: {stats['max_abs']:.3e} J/m²")
    print(f"Relative error mean: {stats['rel_mean']:.2f}%")
    print(f"Relative error std: {stats['rel_std']:.2f}%")
    print()
    
    # Model performance assessment
    print("🎯 MODEL PERFORMANCE")
    print("-" * 20)
    
    r_squared = 1 - np.var(df['residuals']) / np.var(df['E_meas'])
    print(f"R² coefficient: {r_squared:.4f}")
    
    if r_squared > 0.99:
        print("✅ Excellent model agreement")
    elif r_squared > 0.95:
        print("✅ Good model agreement")
    elif r_squared > 0.90:
        print("⚠️ Acceptable model agreement")
    else:
        print("❌ Poor model agreement - investigate")
    print()
    
    # Drift detection
    print("📈 DRIFT DETECTION")
    print("-" * 18)
    
    drift_info = drift_detection(df)
    
    if drift_info['significant_drift']:
        print("❌ Significant drift detected!")
        print(f"   Drift rate: {drift_info['slope']:.2e} J/m²/measurement")
        print(f"   R²: {drift_info['r_squared']:.3f}")
    else:
        print("✅ No significant drift detected")
    print()
    
    # Outlier detection
    print("🔍 OUTLIER DETECTION")
    print("-" * 20)
    
    outliers = outlier_detection(df)
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    if len(outliers) > 0:
        print("Outlier measurements:")
        for idx in outliers.index[:5]:  # Show first 5 outliers
            print(f"  Measurement {idx}: residual = {df.loc[idx, 'residuals']:.2e} J/m²")
        if len(outliers) > 5:
            print(f"  ... and {len(outliers)-5} more")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 18)
    
    if stats['rel_std'] > 10:
        print("⚠️ High measurement variability")
        print("   → Improve experimental control")
        print("   → Check calibration")
    
    if drift_info['significant_drift']:
        print("⚠️ Systematic drift present")
        print("   → Check for environmental changes")
        print("   → Recalibrate instruments")
    
    if len(outliers) > len(df) * 0.1:
        print("⚠️ Many outliers detected")
        print("   → Review experimental procedure")
        print("   → Check for systematic errors")
    
    if r_squared < 0.95:
        print("⚠️ Model agreement could be better")
        print("   → Review theoretical predictions")
        print("   → Consider additional physics")
    
    print()
    
    return {
        'stats': stats,
        'drift_info': drift_info,
        'outliers': outliers,
        'r_squared': r_squared
    }

def save_monitoring_log(df, output_dir='logs'):
    """Save processed data and monitoring results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save processed data
    processed_file = f"{output_dir}/processed_data_{timestamp}.csv"
    df.to_csv(processed_file, index=False)
    
    # Save monitoring summary
    summary_file = f"{output_dir}/monitoring_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Casimir Array Monitoring Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Data points: {len(df)}\n")
        f.write(f"R²: {1 - np.var(df['residuals']) / np.var(df['E_meas']):.4f}\n")
        f.write(f"RMS error: {np.sqrt(np.mean(df['residuals']**2)):.3e} J/m²\n")
    
    print(f"📁 Results saved to {output_dir}/")
    
    return processed_file, summary_file

def plot_residuals(df, save_path=None):
    """Create diagnostic plots for residual analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs predicted
    axes[0,0].scatter(df['E_pred'], df['residuals'], alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Predicted Energy (J/m²)')
    axes[0,0].set_ylabel('Residuals (J/m²)')
    axes[0,0].set_title('Residuals vs Predicted')
    
    # Residual histogram
    axes[0,1].hist(df['residuals'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Residuals (J/m²)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Residual Distribution')
    
    # Time series (if available)
    if 'timestamp' in df.columns:
        axes[1,0].plot(df['timestamp'], df['residuals'], marker='o', markersize=3)
        axes[1,0].axhline(y=0, color='red', linestyle='--')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Residuals (J/m²)')
        axes[1,0].set_title('Residuals vs Time')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].plot(df['residuals'], marker='o', markersize=3)
        axes[1,0].axhline(y=0, color='red', linestyle='--')
        axes[1,0].set_xlabel('Measurement Index')
        axes[1,0].set_ylabel('Residuals (J/m²)')
        axes[1,0].set_title('Residuals vs Measurement Order')
    
    # QQ plot (simplified)
    from scipy import stats
    stats.probplot(df['residuals'], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot (Normal)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Diagnostic plots saved to {save_path}")
    
    plt.show()
    
    return fig

# Example usage
if __name__=='__main__':
    print("🔬 CASIMIR ARRAY DATA RESIDUALS ANALYSIS")
    print("=" * 42)
    print()
    
    # Load or create data
    df = load_measurements('logs/casimir_array_logs.csv')
    
    # Compute predictions
    df = compute_predicted(df)
    
    # Generate monitoring report
    results = real_time_monitoring_report(df)
    
    # Save results
    processed_file, summary_file = save_monitoring_log(df)
    
    # Create diagnostic plots
    try:
        plot_residuals(df, save_path='logs/residual_diagnostics.png')
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    print("=" * 42)
    print("✅ Analysis complete!")
    print(f"📄 Check {summary_file} for summary")
    print(f"📊 Check logs/ directory for detailed results")
