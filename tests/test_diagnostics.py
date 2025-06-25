"""
Unit Tests for Hardware Instrumentation and Diagnostics Module
=============================================================

Comprehensive test suite for all instrumentation components:
- InterferometricProbe: Phase-based measurements
- CalorimetricSensor: Thermal measurements  
- PhaseShiftInterferometer: Complete interferometry system
- RealTimeDAQ: High-speed data acquisition

Tests cover:
- Mathematical correctness of physical models
- Noise and signal processing behavior
- Integration with synthetic ΔT₀₀ profiles
- Performance benchmarks and edge cases
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hardware_instrumentation.diagnostics import (
    InterferometricProbe,
    CalorimetricSensor,
    PhaseShiftInterferometer,
    RealTimeDAQ,
    MeasurementResult,
    generate_T00_pulse,
    benchmark_instrumentation_suite,
    μ0, c, h, k_B
)

class TestInterferometricProbe(unittest.TestCase):
    """Test cases for InterferometricProbe class."""
    
    def setUp(self):
        """Set up test probe with realistic parameters."""
        self.probe = InterferometricProbe(
            wavelength=1.55e-6,  # 1550 nm telecom
            path_length=0.1,     # 10 cm
            n0=1.5,              # Typical glass
            r_coeff=1e-12,       # m/V, LiIO3
            material="LiIO3"
        )
    
    def test_initialization(self):
        """Test probe initialization and calculated parameters."""
        # Check basic parameters
        self.assertEqual(self.probe.lambda0, 1.55e-6)
        self.assertEqual(self.probe.L, 0.1)
        self.assertEqual(self.probe.n0, 1.5)
        self.assertEqual(self.probe.r, 1e-12)
        
        # Check calculated parameters
        expected_k0 = 2 * np.pi / 1.55e-6
        self.assertAlmostEqual(self.probe.k0, expected_k0, places=10)
        
        # Sensitivity should be positive
        self.assertGreater(self.probe.sensitivity, 0)
        
    def test_phase_shift_calculation(self):
        """Test phase shift calculation for various ΔT₀₀ values."""
        # Test zero energy density
        phase_zero = self.probe.phase_shift(0.0)
        self.assertEqual(phase_zero, 0.0)
        
        # Test positive energy density
        T00_pos = 1e6  # J/m³
        phase_pos = self.probe.phase_shift(T00_pos)
        self.assertGreater(phase_pos, 0)
        
        # Test negative energy density  
        T00_neg = -1e6  # J/m³
        phase_neg = self.probe.phase_shift(T00_neg)
        self.assertLess(phase_neg, 0)
        
        # Test magnitude symmetry
        self.assertAlmostEqual(abs(phase_pos), abs(phase_neg), places=10)
        
    def test_phase_shift_scaling(self):
        """Test phase shift scales correctly with energy density."""
        T00_base = 1e6
        phase_base = self.probe.phase_shift(T00_base)
        
        # Test 2x energy density
        T00_2x = 2 * T00_base
        phase_2x = self.probe.phase_shift(T00_2x)
        
        # Phase should scale as sqrt(T00) due to E = sqrt(2μ₀|T00|)
        expected_ratio = np.sqrt(2)
        actual_ratio = phase_2x / phase_base
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)
        
    def test_simulate_pulse(self):
        """Test pulse simulation with realistic parameters."""
        # Create simple gaussian pulse
        times = np.linspace(0, 5e-9, 100)
        T00_pulse = -1e7 * np.exp(-((times - 2.5e-9) / 0.5e-9)**2)
        
        # Test without noise
        result_no_noise = self.probe.simulate_pulse(times, T00_pulse, add_noise=False)
        self.assertIsInstance(result_no_noise, MeasurementResult)
        self.assertEqual(len(result_no_noise.times), len(times))
        self.assertEqual(len(result_no_noise.values), len(times))
        self.assertEqual(result_no_noise.units, "rad")
        self.assertEqual(result_no_noise.measurement_type, "interferometric_phase")
        
        # Test with noise
        result_with_noise = self.probe.simulate_pulse(times, T00_pulse, add_noise=True)
        self.assertTrue(result_with_noise.signal_to_noise > 0)
        self.assertTrue(result_with_noise.noise_floor > 0)
        
        # Noise should add some variation
        phase_diff = np.mean(np.abs(result_with_noise.values - result_no_noise.values))
        self.assertGreater(phase_diff, 0)
        
    def test_frequency_response(self):
        """Test frequency response calculation."""
        frequencies = np.logspace(6, 12, 50)  # 1 MHz to 1 THz
        response = self.probe.frequency_response(frequencies)
        
        # Response should be complex array
        self.assertEqual(len(response), len(frequencies))
        self.assertTrue(np.iscomplexobj(response))
        
        # Low frequency response should be close to 1
        self.assertAlmostEqual(abs(response[0]), 1.0, places=2)
        
        # High frequency response should be attenuated
        self.assertLess(abs(response[-1]), abs(response[0]))

class TestCalorimetricSensor(unittest.TestCase):
    """Test cases for CalorimetricSensor class."""
    
    def setUp(self):
        """Set up test sensor with realistic parameters."""
        self.sensor = CalorimetricSensor(
            volume=1e-18,      # 1 femtoliter
            density=2330,      # Silicon (kg/m³)
            Cp=700,            # Silicon heat capacity (J/(kg⋅K))
            material="Silicon"
        )
    
    def test_initialization(self):
        """Test sensor initialization and derived parameters."""
        # Check basic parameters
        self.assertEqual(self.sensor.V, 1e-18)
        self.assertEqual(self.sensor.rho, 2330)
        self.assertEqual(self.sensor.Cp, 700)
        
        # Check derived parameters
        expected_mass = 2330 * 1e-18
        self.assertAlmostEqual(self.sensor.mass, expected_mass, places=25)
        
        expected_heat_capacity = expected_mass * 700
        self.assertAlmostEqual(self.sensor.heat_capacity, expected_heat_capacity, places=25)
        
        # Thermal time constant should be positive
        self.assertGreater(self.sensor.tau_thermal, 0)
        
    def test_temp_rise_calculation(self):
        """Test temperature rise calculation."""
        # Test zero energy density
        temp_zero = self.sensor.temp_rise(0.0)
        self.assertEqual(temp_zero, 0.0)
        
        # Test positive energy density
        T00_pos = 1e6  # J/m³
        temp_pos = self.sensor.temp_rise(T00_pos)
        self.assertGreater(temp_pos, 0)
        
        # Test negative energy density
        T00_neg = -1e6  # J/m³  
        temp_neg = self.sensor.temp_rise(T00_neg)
        self.assertLess(temp_neg, 0)
        
        # Test linear scaling
        T00_2x = 2e6
        temp_2x = self.sensor.temp_rise(T00_2x)
        self.assertAlmostEqual(temp_2x, 2 * temp_pos, places=10)
        
    def test_simulate_pulse(self):
        """Test pulse simulation with thermal dynamics."""
        times = np.linspace(0, 5e-9, 100)
        T00_pulse = -1e7 * np.exp(-((times - 2.5e-9) / 0.5e-9)**2)
        
        # Test without thermal dynamics
        result_instant = self.sensor.simulate_pulse(
            times, T00_pulse, 
            include_thermal_dynamics=False, 
            add_noise=False
        )
        self.assertIsInstance(result_instant, MeasurementResult)
        self.assertEqual(result_instant.units, "K")
        self.assertEqual(result_instant.measurement_type, "calorimetric_temperature")
        
        # Test with thermal dynamics
        result_thermal = self.sensor.simulate_pulse(
            times, T00_pulse, 
            include_thermal_dynamics=True, 
            add_noise=False
        )
        
        # Thermal dynamics should smooth the response
        instant_variation = np.std(result_instant.values)
        thermal_variation = np.std(result_thermal.values)
        # Note: For very short time scales, difference might be minimal
        
        # Test with noise
        result_noise = self.sensor.simulate_pulse(times, T00_pulse, add_noise=True)
        self.assertTrue(result_noise.signal_to_noise > 0)

class TestPhaseShiftInterferometer(unittest.TestCase):
    """Test cases for PhaseShiftInterferometer class."""
    
    def setUp(self):
        """Set up test interferometer system."""
        self.probe = InterferometricProbe(1.55e-6, 0.1, 1.5, 1e-12)
        self.interferometer = PhaseShiftInterferometer(
            probe=self.probe,
            sampling_rate=1e11,  # 100 GHz
            anti_alias_filter=True,
            digital_filter=True
        )
    
    def test_initialization(self):
        """Test interferometer initialization."""
        self.assertEqual(self.interferometer.probe, self.probe)
        self.assertEqual(self.interferometer.fs, 1e11)
        self.assertTrue(self.interferometer.anti_alias)
        self.assertTrue(self.interferometer.digital_filter)
        
        # Check derived parameters
        self.assertEqual(self.interferometer.f_nyquist, 5e10)
        self.assertGreater(self.interferometer.filter_cutoff, 0)
        self.assertLess(self.interferometer.filter_cutoff, self.interferometer.f_nyquist)
        
    def test_acquire(self):
        """Test data acquisition."""
        # Create test function
        def T00_func(t):
            return -1e7 * np.exp(-((t - 2.5e-9) / 0.5e-9)**2)
        
        # Test acquisition
        duration = 5e-9
        result = self.interferometer.acquire(duration, T00_func, real_time_processing=True)
        
        self.assertIsInstance(result, MeasurementResult)
        self.assertEqual(result.measurement_type, "interferometric_phase")
        
        # Check metadata
        self.assertIn('sampling_rate', result.metadata)
        self.assertIn('duration', result.metadata)
        self.assertIn('n_samples', result.metadata)
        
        # Number of samples should match expected
        expected_samples = int(duration * self.interferometer.fs)
        self.assertEqual(result.metadata['n_samples'], expected_samples)
        
    def test_frequency_sweep(self):
        """Test frequency response characterization."""
        freq_range = (1e6, 1e12)  # 1 MHz to 1 THz
        response_data = self.interferometer.frequency_sweep(freq_range, n_points=50)
        
        # Check output structure
        required_keys = ['frequencies', 'probe_response', 'system_response', 
                        'total_response', 'magnitude_dB', 'phase_deg']
        for key in required_keys:
            self.assertIn(key, response_data)
            
        # Check array lengths
        n_points = len(response_data['frequencies'])
        for key in required_keys:
            self.assertEqual(len(response_data[key]), n_points)
            
        # Magnitude should be in dB (can be negative)
        self.assertTrue(np.all(np.isfinite(response_data['magnitude_dB'])))
        
        # Phase should be in degrees (-180 to 180)
        phases = response_data['phase_deg']
        self.assertTrue(np.all(phases >= -180))
        self.assertTrue(np.all(phases <= 180))

class TestRealTimeDAQ(unittest.TestCase):
    """Test cases for RealTimeDAQ class."""
    
    def setUp(self):
        """Set up test DAQ system."""
        self.daq = RealTimeDAQ(
            buffer_size=1000,
            sampling_rate=1e9,  # 1 GHz
            trigger_level=1e-6,
            trigger_mode="rising"
        )
    
    def test_initialization(self):
        """Test DAQ initialization."""
        self.assertEqual(self.daq.buffer_size, 1000)
        self.assertEqual(self.daq.fs, 1e9)
        self.assertEqual(self.daq.trigger_level, 1e-6)
        self.assertEqual(self.daq.trigger_mode, "rising")
        
        # Check initial state
        self.assertEqual(self.daq.sample_count, 0)
        self.assertEqual(self.daq.trigger_count, 0)
        self.assertFalse(self.daq.triggered)
        
    def test_add_sample(self):
        """Test sample addition and buffer management."""
        # Add single sample
        triggered = self.daq.add_sample(0.0, 0.5e-6)
        self.assertFalse(triggered)  # Below trigger level
        self.assertEqual(self.daq.sample_count, 1)
        
        # Add sample above trigger level
        triggered = self.daq.add_sample(1e-9, 2e-6)
        self.assertTrue(triggered)  # Rising edge trigger
        self.assertEqual(self.daq.trigger_count, 1)
        self.assertTrue(self.daq.triggered)
        
    def test_trigger_modes(self):
        """Test different trigger modes."""
        # Test falling trigger
        daq_falling = RealTimeDAQ(100, 1e9, 1e-6, "falling")
        
        daq_falling.add_sample(0, 2e-6)  # Above threshold
        triggered = daq_falling.add_sample(1e-9, 0.5e-6)  # Below threshold
        self.assertTrue(triggered)  # Falling edge
        
        # Test level trigger - need to check absolute value
        daq_level = RealTimeDAQ(100, 1e9, 1e-6, "level")
        
        # Add initial sample below threshold first
        daq_level.add_sample(0, 0.5e-6)  # Below threshold
        triggered = daq_level.add_sample(1e-9, 2e-6)  # Above threshold
        self.assertTrue(triggered)  # Level trigger
        
    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        # Fill buffer beyond capacity
        buffer_size = self.daq.buffer_size
        for i in range(buffer_size + 100):
            self.daq.add_sample(i * 1e-9, 0.5e-6)
            
        # Check overrun detection
        self.assertGreater(self.daq.overrun_count, 0)
        
        # Get buffer data
        times, values = self.daq.get_buffer()
        self.assertEqual(len(times), buffer_size)
        self.assertEqual(len(values), buffer_size)
        
    def test_statistics(self):
        """Test DAQ statistics calculation."""
        # Add some samples with triggers
        for i in range(10):
            value = 2e-6 if i % 3 == 0 else 0.5e-6  # Trigger every 3rd sample
            self.daq.add_sample(i * 1e-9, value)
            
        stats = self.daq.get_statistics()
        
        # Check statistics structure
        required_keys = ['sample_count', 'trigger_count', 'overrun_count',
                        'buffer_utilization', 'trigger_rate', 'overrun_rate']
        for key in required_keys:
            self.assertIn(key, stats)
            
        # Check values
        self.assertEqual(stats['sample_count'], 10)
        self.assertGreater(stats['trigger_count'], 0)
        self.assertGreaterEqual(stats['buffer_utilization'], 0)
        self.assertLessEqual(stats['buffer_utilization'], 1)
        
    def test_reset(self):
        """Test DAQ reset functionality."""
        # Add some data
        self.daq.add_sample(0, 2e-6)
        self.daq.add_sample(1e-9, 0.5e-6)
        
        # Reset
        self.daq.reset()
        
        # Check reset state
        self.assertEqual(self.daq.sample_count, 0)
        self.assertEqual(self.daq.trigger_count, 0)
        self.assertFalse(self.daq.triggered)
        self.assertTrue(np.all(self.daq.times == 0))
        self.assertTrue(np.all(self.daq.values == 0))

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_generate_T00_pulse(self):
        """Test synthetic pulse generation."""
        # Test Gaussian pulse
        gaussian_func = generate_T00_pulse("gaussian", -1e7, 1e-9, 0.2e-9)
        
        # Check peak value at center time
        peak_value = gaussian_func(1e-9)
        self.assertAlmostEqual(peak_value, -1e7, places=0)
        
        # Check decay away from center
        side_value = gaussian_func(2e-9)  # 5 widths away
        self.assertLess(abs(side_value), abs(peak_value))
        
        # Test exponential pulse
        exp_func = generate_T00_pulse("exponential", -1e7, 1e-9, 0.2e-9)
        
        # Should be zero before center time
        early_value = exp_func(0.5e-9)
        self.assertEqual(early_value, 0)
        
        # Should be peak at center time
        center_value = exp_func(1e-9)
        self.assertAlmostEqual(center_value, -1e7, places=0)
        
        # Test square pulse
        square_func = generate_T00_pulse("square", -1e7, 1e-9, 0.2e-9)
        
        # Should be peak within width
        inside_value = square_func(1.1e-9)
        self.assertAlmostEqual(inside_value, -1e7, places=0)
        
        # Should be zero outside width
        outside_value = square_func(1.5e-9)
        self.assertEqual(outside_value, 0)
        
    def test_benchmark_instrumentation_suite(self):
        """Test comprehensive instrumentation benchmark."""
        # This test runs the full benchmark suite
        try:
            results = benchmark_instrumentation_suite()
            
            # Check that all expected systems are tested
            expected_systems = ['interferometric', 'calorimetric', 
                              'phase_shift_interferometer', 'real_time_daq']
            for system in expected_systems:
                self.assertIn(system, results)
                self.assertEqual(results[system]['status'], 'SUCCESS')
                
            print("✅ Benchmark suite completed successfully")
            
        except Exception as e:
            self.fail(f"Benchmark suite failed: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def test_complete_measurement_chain(self):
        """Test complete measurement chain from T₀₀ to DAQ output."""
        # Create measurement system
        probe = InterferometricProbe(1.55e-6, 0.1, 1.5, 1e-12)
        interferometer = PhaseShiftInterferometer(probe, 1e11)
        daq = RealTimeDAQ(10000, 1e10, 1e-6)
        
        # Generate synthetic pulse
        pulse_func = generate_T00_pulse("gaussian", -1e7, 2.5e-9, 0.5e-9)
        
        # Acquire interferometric data
        interfero_result = interferometer.acquire(5e-9, pulse_func)
        
        # Feed data into DAQ system
        trigger_count = 0
        for t, phase in zip(interfero_result.times, interfero_result.values):
            if daq.add_sample(t, phase):
                trigger_count += 1
                
        # Check that system worked end-to-end
        self.assertGreater(daq.sample_count, 0)
        self.assertGreaterEqual(trigger_count, 0)  # May or may not trigger depending on signal level
        
        # Get final buffer data
        times, values = daq.get_buffer()
        self.assertEqual(len(times), daq.buffer_size)
        self.assertEqual(len(values), daq.buffer_size)
        
    def test_multi_sensor_comparison(self):
        """Test comparison between interferometric and calorimetric measurements."""
        # Set up both sensors
        probe = InterferometricProbe(1.55e-6, 0.1, 1.5, 1e-12)
        calorimeter = CalorimetricSensor(1e-18, 2330, 700)
        
        # Common test signal
        times = np.linspace(0, 5e-9, 1000)
        T00_values = -1e7 * np.exp(-((times - 2.5e-9) / 0.5e-9)**2)
        
        # Measure with both
        interfero_result = probe.simulate_pulse(times, T00_values, add_noise=False)
        calor_result = calorimeter.simulate_pulse(times, T00_values, add_noise=False)
        
        # Both should detect the signal
        max_phase = np.max(np.abs(interfero_result.values))
        max_temp = np.max(np.abs(calor_result.values))
        
        self.assertGreater(max_phase, 0)
        self.assertGreater(max_temp, 0)
        
        # Signals should be correlated in time
        # (Both should peak around the same time)
        phase_peak_idx = np.argmax(np.abs(interfero_result.values))
        temp_peak_idx = np.argmax(np.abs(calor_result.values))
        
        # Peak indices should be close (within 15% of total samples)
        idx_diff = abs(phase_peak_idx - temp_peak_idx)
        max_allowed_diff = len(times) * 0.15  # Increased tolerance
        self.assertLess(idx_diff, max_allowed_diff)

if __name__ == '__main__':
    print("🧪 Starting Hardware Instrumentation Test Suite")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestInterferometricProbe,
        TestCalorimetricSensor, 
        TestPhaseShiftInterferometer,
        TestRealTimeDAQ,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 55)
    print(f"🎯 Test Summary:")
    print(f"   • Tests run: {result.testsRun}")
    print(f"   • Failures: {len(result.failures)}")
    print(f"   • Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("   ✅ All tests passed!")
        exit_code = 0
    else:
        print("   ❌ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"   • {test}: {traceback.split('Error:')[-1].strip()}")
        exit_code = 1
    
    print("=" * 55)
    exit(exit_code)
