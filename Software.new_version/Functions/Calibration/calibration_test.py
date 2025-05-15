import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tempfile
import spectral
import spectral.io.envi as envi

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Now import will work regardless of where the script is run from
from Functions.Calibration.calibrate import calibration, apply_calibration_model

def create_mock_hsi(shape, wavelengths=None, filename=None):
    """Create a mock hyperspectral image for testing."""
    # Create data with some spectral patterns
    data = np.zeros(shape, dtype=np.float32)
    
    # Add some spectral patterns
    for band in range(shape[2]):
        # Create a pattern that varies with wavelength
        scale = 0.5 * np.sin(band / shape[2] * 2 * np.pi) + 0.5
        data[:, :, band] = scale * np.random.rand(shape[0], shape[1]) * 100
    
    # Create metadata
    if wavelengths is None:
        # Generate some mock wavelengths
        wavelengths = [400 + i * 10 for i in range(shape[2])]
    
    metadata = {
        'wavelength': wavelengths,
        'wavelength units': 'nm',
        'data type': '4',  # float32
        'interleave': 'bil',
        'lines': shape[0],
        'samples': shape[1],
        'bands': shape[2],
    }
    
    if filename:
        # Save the file
        header_file = filename + '.hdr'
        bil_file = filename + '.bil'
        
        envi.save_image(header_file, data, dtype=np.float32, interleave='bil', 
                       ext='bil', force=True, metadata=metadata)
        
        # Return the opened file
        return envi.open(header_file, bil_file)
    else:
        # Just return the data and metadata
        return data, metadata

def test_calibration():
    """Test the calibration function."""
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Created temporary directory: {temp_dir}")
        
        # Define shapes for test data
        shape = (10, 10, 5)  # small size for quick testing (lines, samples, bands)
        
        # Create mock wavelengths (400-440nm)
        wavelengths = [400, 410, 420, 430, 440]
        
        # Create mock data files
        dark_path = os.path.join(temp_dir, 'dark')
        ref_path = os.path.join(temp_dir, 'ref')
        input_path = os.path.join(temp_dir, 'input')
        output_path = os.path.join(temp_dir, 'output')
        
        # Create mock HSI files
        dark_hsi = create_mock_hsi(shape, wavelengths, dark_path)
        ref_hsi = create_mock_hsi(shape, wavelengths, ref_path)
        input_hsi = create_mock_hsi(shape, wavelengths, input_path)
        
        print("Created mock HSI files.")
        
        # Test the calibration function
        print("Testing calibration function...")
        try:
            calibrated_hsi = calibration(dark_hsi, ref_hsi, input_hsi, output_path)
            print("Calibration succeeded!")
            print(f"Calibrated HSI shape: {calibrated_hsi.shape}")
            
            # Verify the calibrated data
            calibrated_data = calibrated_hsi.read_bands(range(shape[2]))
            print(f"Calibrated data min: {np.min(calibrated_data)}, max: {np.max(calibrated_data)}")
            
            # Check if output files were created
            output_hdr = output_path + '.hdr'
            output_bil = output_path + '.bil'
            if os.path.exists(output_hdr) and os.path.exists(output_bil):
                print(f"Output files created successfully: {output_hdr} and {output_bil}")
            else:
                print("Output files were not created.")
            
            # Test the apply_calibration_model function
            print("\nTesting apply_calibration_model function...")
            try:
                # Use the output of calibration as our model
                model_path = output_bil
                new_input_path = os.path.join(temp_dir, 'new_input')
                new_output_path = os.path.join(temp_dir, 'new_output')
                
                # Create a new input file
                new_input_hsi = create_mock_hsi(shape, wavelengths, new_input_path)
                
                # Apply the model
                applied_hsi = apply_calibration_model(new_input_hsi, model_path, new_output_path)
                print("Model application succeeded!")
                print(f"Applied HSI shape: {applied_hsi.shape}")
                
                # Verify the applied data
                applied_data = applied_hsi.read_bands(range(shape[2]))
                print(f"Applied data min: {np.min(applied_data)}, max: {np.max(applied_data)}")
                
                # Check if output files were created
                new_output_hdr = new_output_path + '.hdr'
                new_output_bil = new_output_path + '.bil'
                if os.path.exists(new_output_hdr) and os.path.exists(new_output_bil):
                    print(f"New output files created successfully: {new_output_hdr} and {new_output_bil}")
                else:
                    print("New output files were not created.")
                
            except Exception as e:
                print(f"Error in apply_calibration_model: {str(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in calibration: {str(e)}")
            import traceback
            traceback.print_exc()
            
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def visualize_results(original, calibrated):
    """Visualize original and calibrated data."""
    # Create a RGB visualization for both
    # For this example, we'll just use the first 3 bands for RGB
    bands_rgb = [0, 1, 2]
    
    # Get RGB for original
    original_rgb = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
    for i, band in enumerate(bands_rgb):
        # Scale to 0-255
        band_data = original[:, :, band]
        min_val = np.min(band_data)
        max_val = np.max(band_data)
        scaled = (band_data - min_val) / (max_val - min_val) * 255
        original_rgb[:, :, i] = scaled.astype(np.uint8)
    
    # Get RGB for calibrated
    calibrated_rgb = np.zeros((calibrated.shape[0], calibrated.shape[1], 3), dtype=np.uint8)
    for i, band in enumerate(bands_rgb):
        # Scale to 0-255
        band_data = calibrated[:, :, band]
        min_val = np.min(band_data)
        max_val = np.max(band_data)
        scaled = (band_data - min_val) / (max_val - min_val) * 255
        calibrated_rgb[:, :, i] = scaled.astype(np.uint8)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_rgb)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2.imshow(calibrated_rgb)
    ax2.set_title("Calibrated")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function."""
    print("Testing calibration functionality...")
    test_calibration()
    print("All tests completed.")

if __name__ == "__main__":
    main()