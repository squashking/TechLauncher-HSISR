# Hyperspectral Image Classification GUI

This project implements a PyQt6-based graphical user interface (GUI) for hyperspectral image classification. The classification component uses a Gaussian classifier to classify hyperspectral image data based on ground truth masks. The interface allows users to load image data, configure classification settings, and visualize classification results.

## Features

- **Load Hyperspectral Images**: Select `.bil` format hyperspectral images along with their corresponding `.hdr` metadata files.
- **Ground Truth Input**: Provide a ground truth mask image (`.jpg`) for supervised classification.
- **Classification Methods**: Perform supervised classification using the Gaussian Classifier.
- **Visualize Results**: View classified output images directly within the GUI.

## Getting Started

### Prerequisites

- Python 3.x
- PyQt6
- Spectral Python (SPy)
- Matplotlib
- PIL (Pillow)

Install the required packages:
```bash
pip install pyqt6 spectral matplotlib pillow
```

### Running the Application

1. **Clone the Repository**:
```bash
git clone https://github.com/squashking/TechLauncher-HSISR.git
cd TechLauncher-HSISR
```

2. **Run the GUI**:
```bash
python main_window.py
```

### Using the GUI

1. **Loading Hyperspectral Image**:
   2. In the **Classification** tab, enter the path to your `.bil` hyperspectral image file or use the "Browse" button to select it.
   3. Ensure the corresponding `.hdr` file is in the same directory.

2. **Providing Ground Truth**:
   2. Enter or browse for the path to the ground truth mask image (`.jpg` format).

3. **Classification**:
   2. Select **Supervised** classification mode.
   3. Choose the **Gaussian Classifier** (default).
   4. Click **Classify** to perform the classification.

4. **Viewing Results**:
   2. After classification, the output image will be displayed in the window. Results are also saved as `gtresults.png` in the `result` folder.
   
      <img src="/Users/zhenwang/Library/Application Support/typora-user-images/image-20240906145714465.png" alt="image-20240906145714465" style="zoom: 33%;" />

## Directory Structure

```
.
├── One_sample/
│   ├── 2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.bil
│   ├── 2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.hdr
│   ├── 2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1_mask.jpg
├── result/  # This folder will store the classification output
│   └── gtresults.png
├── hyperspectral_classifier.py  # Classification logic and functionality
├── main_window.py  # GUI implementation
├── README.md
```

## Notes

- Ensure the `.bil` and `.hdr` files are located in the same directory.
- The default ground truth file is set to `One_sample/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1_mask.jpg`. This can be changed within the GUI.
