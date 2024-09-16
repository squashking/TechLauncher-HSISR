import numpy as np
from spectral import envi, kmeans

# Function to calculate NDVI
def calculate_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    denominator = nir + red
    denominator[denominator == 0] = np.nan  # Avoid division by zero
    ndvi = (nir - red) / denominator
    return np.nan_to_num(ndvi, nan=0.0)  # Replace NaN with 0.0

# Function to find Red and NIR bands
def find_Red_NIR_bands(listWavelength):
    R_wavelength = 682.5  # Red (625+740)/2
    NIR_wavelength = 850  # NIR

    rFound = nirFound = False
    rPreDifference = nirPreDifference = float('inf')  # previously calculated difference
    rIndex = nirIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:
                rPreDifference = difference
            else:
                rIndex = i - 1
                rFound = True

        if not nirFound:
            difference = abs(value - NIR_wavelength)
            if difference < nirPreDifference:
                nirPreDifference = difference
            else:
                nirIndex = i - 1
                nirFound = True

    return (rIndex, nirIndex)

# Function to load HSI data and perform clustering
def load_and_process_hsi_data(hdr_file, bil_file, k=20, max_iterations=15):
    img = envi.open(hdr_file, image=bil_file)
    data = img.load()

    # Perform K-Means clustering
    m, c = kmeans(data, k, max_iterations)
    cluster_map = m.reshape(data.shape[:-1])  # Reshape to 2D map (rows, cols)

    # Extract wavelength data from the metadata
    wavelengths = [float(w) for w in img.metadata['wavelength']]

    # Find Red and NIR bands
    red_band_index, nir_band_index = find_Red_NIR_bands(wavelengths)
    red_band = data[:, :, red_band_index]
    nir_band = data[:, :, nir_band_index]

    # Calculate NDVI
    ndvi = calculate_ndvi(nir_band, red_band)

    return cluster_map, ndvi, img

