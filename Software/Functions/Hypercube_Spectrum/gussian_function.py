import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Function to find RGB band weights based on Gaussian distribution around target wavelengths
def find_RGB_bands(listWavelength):
    R_wavelength, G_wavelength, B_wavelength = 650, 550, 450  # Target center wavelengths for R, G, B
    sigma = 10  # Standard deviation for Gaussian weighting

    # Compute normalized Gaussian weights centered at target wavelength
    def gaussian_weights(target, wavelengths):
        weights = norm.pdf(wavelengths, target, sigma)  # Gaussian probability density
        return weights / np.sum(weights)  # Normalize the weights to sum to 1

    listWavelength = np.array(listWavelength)
    R_weights = gaussian_weights(R_wavelength, listWavelength)
    G_weights = gaussian_weights(G_wavelength, listWavelength)
    B_weights = gaussian_weights(B_wavelength, listWavelength)

    return R_weights, G_weights, B_weights  # Return RGB weights

# Function to read PSI-style header file and extract wavelengths
def read_PSI_header(headerPath):
    wavelengths = []
    with open(headerPath, "r") as file:
        for line in file:
            if line.startswith("WAVELENGTHS"):  # Find the line with wavelength data
                # Extract all values after the keyword and convert them to float
                wavelengths = list(map(float, line.strip().split(" ")[1:]))
    return {"wavelengths": wavelengths}  # Return wavelengths as metadata dictionary

# Function to create a valid ENVI header for use with spectral.io
def create_envi_header(output_path, metadata, shape):
    header_path = output_path + ".hdr"  # Header file name
    with open(header_path, "w") as header:
        header.write("ENVI\n")
        header.write(f"samples = {shape[1]}\n")     # Number of columns (width)
        header.write(f"lines = {shape[0]}\n")       # Number of rows (height)
        header.write(f"bands = {shape[2]}\n")       # Number of spectral bands
        header.write("interleave = bsq\n")          # Band Sequential format
        header.write("byte order = 0\n")            # Byte order (0 = little-endian)
        # Write the wavelengths in ENVI header format
        header.write(f"wavelength = {{{', '.join(map(str, metadata['wavelengths']))}}}\n")

# Function to load image using its path and header
def load_image(imagePath, headerPath):
    metadata = read_PSI_header(headerPath)  # Read metadata from PSI header
    # Create ENVI header with placeholder shape (100, 100, bands) for demonstration
    create_envi_header(imagePath, metadata, (100, 100, len(metadata["wavelengths"])))
    return envi.open(imagePath + ".hdr", imagePath)  # Load image using spectral.envi

# Normalize hyperspectral data to [0, 1] range
def normalize_data(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Display an RGB composite from hyperspectral cube using weighted band combinations
def show_cube(image, metadata, save_path=""):
    wavelengths = np.array(metadata["wavelengths"])  # Convert wavelength list to NumPy array
    R_weights, G_weights, B_weights = find_RGB_bands(wavelengths)  # Get RGB weights
    # The rest of the visualization would go here (e.g., weighted sum of bands for RGB image)
