import sys
import os
import numpy as np
import spectral.io.envi as envi

def calibration(dark_hsi, ref_hsi, input_hsi, output_filename=None):
    """
    Calibrate a hyperspectral image using dark and reference images.
    
    Parameters:
    -----------
    dark_hsi : spectral.SpyFile
        Dark reference image
    ref_hsi : spectral.SpyFile
        White reference image
    input_hsi : spectral.SpyFile
        Input image to calibrate
    output_filename : str, optional
        Output file name without extension (default: None)
        
    Returns:
    --------
    spectral.SpyFile
        Calibrated image
    """
    assert dark_hsi.shape == ref_hsi.shape
    
    # Calculate subtracting and dividing factors
    subt = np.repeat(
        np.average(dark_hsi[:, :, :], axis=0, keepdims=True), 
        input_hsi.shape[0], axis=0
    )
    
    divisor = np.repeat(
        np.average(ref_hsi[:, :, :] - dark_hsi[:, :, :], axis=0, keepdims=True), 
        input_hsi.shape[0], axis=0
    )
    
    # Apply calibration formula and clip to valid range
    result = np.clip(
        ((input_hsi[:, :, :] - subt) / divisor) * 255, 
        0, 255
    )
    
    # Set output file paths
    if output_filename is None:
        output_hdr = "calibration.hdr"
        output_bil = "calibration.bil"
    else:
        # Make sure the directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_hdr = output_filename + ".hdr"
        output_bil = output_filename + ".bil"
    
    # Save calibrated image
    envi.save_image(output_hdr, result, dtype=np.uint16, interleave="bil", 
                   ext="bil", force=True, metadata=dark_hsi.metadata)
    
    # Open and return the calibrated image
    output_hsi = envi.open(output_hdr, output_bil)
    
    return output_hsi

def apply_calibration_model(input_hsi, model_path, output_filename=None):
    """
    Apply a pre-existing calibration model to an input image.
    
    Parameters:
    -----------
    input_hsi : spectral.SpyFile
        Input image to calibrate
    model_path : str
        Path to the calibration model file (.bil)
    output_filename : str, optional
        Output file name without extension (default: None)
        
    Returns:
    --------
    spectral.SpyFile
        Calibrated image
    """
    # Extract header path from model path
    model_hdr = model_path.replace(".bil", ".hdr")
    
    if not os.path.exists(model_path) or not os.path.exists(model_hdr):
        raise FileNotFoundError(f"Model file or header not found: {model_path}")
    
    # Load the model
    model = envi.open(model_hdr, model_path)
    
    # Apply model to input image
    # This is a simplified example - you would need to define how
    # to actually apply your calibration model to the input image
    # based on your specific calibration approach
    
    # Example application (may need adjustment based on your specific needs):
    result = np.clip(input_hsi[:, :, :] * model[:, :, :], 0, 255)
    
    # Set output file paths
    if output_filename is None:
        output_hdr = "calibrated.hdr"
        output_bil = "calibrated.bil"
    else:
        # Make sure the directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_hdr = output_filename + ".hdr"
        output_bil = output_filename + ".bil"
    
    # Save calibrated image
    envi.save_image(output_hdr, result, dtype=np.uint16, interleave="bil", 
                   ext="bil", force=True, metadata=input_hsi.metadata)
    
    # Open and return the calibrated image
    output_hsi = envi.open(output_hdr, output_bil)
    
    return output_hsi