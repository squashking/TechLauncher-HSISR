import unittest
import os
import sys
import numpy as np
import tempfile
import shutil
import spectral.io.envi as envi

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Now import will work regardless of where the script is run from
from Functions.Calibration.calibrate import calibration, apply_calibration_model

class MockSpyFile:
    """A mock SpyFile class for testing."""
    def __init__(self, data, metadata=None):
        self.data = data
        self.shape = data.shape
        self.metadata = metadata or {}
        
    def read_bands(self, bands):
        if isinstance(bands, range):
            return self.data[:, :, bands]
        else:
            return self.data[:, :, bands]

class CalibrationTest(unittest.TestCase):
    """Unit tests for calibration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Define shapes for test data
        self.shape = (5, 5, 3)  # small size for quick testing
        
        # Create test data with controlled values
        self.dark_data = np.ones(self.shape) * 10  # Dark reference (constant value)
        self.ref_data = np.ones(self.shape) * 200  # White reference (constant value)
        self.input_data = np.ones(self.shape) * 100  # Input image (constant value)
        
        # Create extreme test data
        self.zero_data = np.zeros(self.shape)  # All zeros
        self.high_data = np.ones(self.shape) * 1000  # Very high values
        
        # Create metadata for the mock files
        self.metadata = {
            'wavelength': [400, 500, 600],
            'wavelength units': 'nm',
            'data type': '4',  # float32
            'interleave': 'bil',
            'lines': self.shape[0],
            'samples': self.shape[1],
            'bands': self.shape[2],
        }
        
        # Create mock SpyFile objects
        self.dark_hsi = MockSpyFile(self.dark_data, self.metadata.copy())
        self.ref_hsi = MockSpyFile(self.ref_data, self.metadata.copy())
        self.input_hsi = MockSpyFile(self.input_data, self.metadata.copy())
        self.zero_hsi = MockSpyFile(self.zero_data, self.metadata.copy())
        self.high_hsi = MockSpyFile(self.high_data, self.metadata.copy())
        
        # Save the mock files to disk for testing with envi functions
        def save_mock_hsi(data, filename):
            """Save mock HSI data to disk."""
            header_file = os.path.join(self.temp_dir, filename + '.hdr')
            bil_file = os.path.join(self.temp_dir, filename + '.bil')
            
            envi.save_image(header_file, data, dtype=np.float32, interleave='bil', 
                           ext='bil', force=True, metadata=self.metadata)
            
            return envi.open(header_file, bil_file)
        
        self.dark_file = save_mock_hsi(self.dark_data, 'dark')
        self.ref_file = save_mock_hsi(self.ref_data, 'ref')
        self.input_file = save_mock_hsi(self.input_data, 'input')
        self.zero_file = save_mock_hsi(self.zero_data, 'zero')
        self.high_file = save_mock_hsi(self.high_data, 'high')
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_calibration_formula_manually(self):
        """Test the calibration formula manually to verify correctness."""
        # Extract the calibration formula from the function:
        # result = ((input_hsi - subt) / divisor) * 255
        # where subt = avg(dark_hsi) and divisor = avg(ref_hsi - dark_hsi)
        
        # Calculate expected values
        subt = np.average(self.dark_data, axis=0, keepdims=True)
        subt = np.repeat(subt, self.shape[0], axis=0)
        
        divisor = np.average(self.ref_data - self.dark_data, axis=0, keepdims=True)
        divisor = np.repeat(divisor, self.shape[0], axis=0)
        
        expected = ((self.input_data - subt) / divisor) * 255
        expected = np.clip(expected, 0, 255)
        
        # Since we're using constant values, we can simplify:
        # subt = 10 (dark value)
        # divisor = 190 (ref - dark = 200 - 10)
        # result = ((100 - 10) / 190) * 255 = (90 / 190) * 255 ≈ 120.79
        simplified_expected = np.ones(self.shape) * ((100 - 10) / 190) * 255
        
        # Check if our manual calculation matches our simplified expectation
        np.testing.assert_allclose(expected, simplified_expected, rtol=1e-5)
        
        # Now run the actual calibration function
        output_path = os.path.join(self.temp_dir, 'output')
        calibrated_hsi = calibration(self.dark_file, self.ref_file, self.input_file, output_path)
        
        # Read the calibrated data
        calibrated_data = calibrated_hsi.read_bands(range(self.shape[2]))
        
        # The calibration function saves as uint16, which means values are rounded to integers
        # We should account for this rounding in our test
        print(f"Expected (float): {expected[0, 0, 0]}, Actual (integer): {calibrated_data[0, 0, 0]}")
        
        # Use a much more relaxed tolerance to account for rounding and dtype conversion
        # Allow for a difference of 1 in integer values (which is about a 1% difference for values around 120)
        np.testing.assert_allclose(calibrated_data, expected, rtol=0.01, atol=1.0)
        
        print("Calibration formula test passed!")
    
    def test_calibration_with_edge_cases(self):
        """Test calibration with edge cases."""
        # Test with zero dark and input (should result in zeros)
        output_path = os.path.join(self.temp_dir, 'edge_case1')
        calibrated_hsi = calibration(self.zero_file, self.ref_file, self.zero_file, output_path)
        calibrated_data = calibrated_hsi.read_bands(range(self.shape[2]))
        # Since we expect all zeros, we can use a small absolute tolerance
        np.testing.assert_allclose(calibrated_data, np.zeros(self.shape), atol=0.1)
        
        # Test with equal dark and reference (should result in divide by zero -> clipped to 0)
        # This should be handled gracefully by the clipping in the function
        output_path = os.path.join(self.temp_dir, 'edge_case2')
        try:
            calibrated_hsi = calibration(self.dark_file, self.dark_file, self.input_file, output_path)
            calibrated_data = calibrated_hsi.read_bands(range(self.shape[2]))
            # With equal dark and ref, divisor becomes 0, resulting in NaN or Inf
            # But the clipping should handle this and set to 0 or 255
            self.assertTrue(np.all(np.isfinite(calibrated_data)))
            print("Edge case with equal dark and reference handled correctly.")
        except Exception as e:
            print(f"Edge case with equal dark and reference failed: {str(e)}")
            raise
        
        # Test with very high values
        output_path = os.path.join(self.temp_dir, 'edge_case3')
        calibrated_hsi = calibration(self.dark_file, self.high_file, self.input_file, output_path)
        calibrated_data = calibrated_hsi.read_bands(range(self.shape[2]))
        # Expected calculation: ((100 - 10) / (1000 - 10)) * 255 ≈ 23.18
        expected = np.ones(self.shape) * ((100 - 10) / (1000 - 10)) * 255
        
        # Print values for comparison
        print(f"High value test - Expected (float): {expected[0, 0, 0]}, Actual (integer): {calibrated_data[0, 0, 0]}")
        
        # Use more relaxed tolerance for integer rounding
        np.testing.assert_allclose(calibrated_data, expected, rtol=0.01, atol=1.0)
        
        print("Edge case tests passed!")
    
    def test_apply_calibration_model(self):
        """Test the apply_calibration_model function."""
        # First, create a calibration model
        model_path = os.path.join(self.temp_dir, 'model')
        calibrated_hsi = calibration(self.dark_file, self.ref_file, self.input_file, model_path)
        
        # Then apply it to a new input
        output_path = os.path.join(self.temp_dir, 'model_applied')
        try:
            applied_hsi = apply_calibration_model(self.high_file, model_path + '.bil', output_path)
            applied_data = applied_hsi.read_bands(range(self.shape[2]))
            
            # Verify output files were created
            self.assertTrue(os.path.exists(output_path + '.hdr'))
            self.assertTrue(os.path.exists(output_path + '.bil'))
            
            # Check shape
            self.assertEqual(applied_hsi.shape, self.shape)
            
            print("Apply calibration model test passed!")
        except Exception as e:
            print(f"Apply calibration model test failed: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main()