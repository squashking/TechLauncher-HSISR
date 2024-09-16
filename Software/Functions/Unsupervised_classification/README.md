# Hyperspectral Image Processing and NDVI Visualization

This project processes hyperspectral images (HSI) to perform k-means clustering and calculate the Normalized Difference Vegetation Index (NDVI). The results are displayed in a PyQt-based GUI where users can interactively view the NDVI values for different regions of the image.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Function Definitions](#function-definitions)
    - [calculate_ndvi](#calculate_ndvi)
    - [find_Red_NIR_bands](#find_red_nir_bands)
- [Loading Hyperspectral Data](#loading-hyperspectral-data)
- [K-Means Clustering](#k-means-clustering)
- [PyQt GUI Implementation](#pyqt-gui-implementation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Introduction

This script is designed to:
1. Load and process hyperspectral images.
2. Perform k-means clustering on the spectral data.
3. Calculate the NDVI using the Red and Near-Infrared (NIR) bands.
4. Display the processed image and allow users to interactively explore the NDVI values of different regions using a graphical user interface (GUI).

## Dependencies

Ensure that the following Python packages are installed:

- `numpy`
- `matplotlib`
- `spectral`
- `PyQt6`

Install these packages using pip:

```bash
pip install numpy matplotlib spectral PyQt6
