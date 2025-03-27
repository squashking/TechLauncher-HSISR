# unsupervised_classification.py

import numpy as np
from spectral import kmeans
import apriltag
import cv2

def calculate_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    denominator = nir + red
    denominator[denominator == 0] = np.nan  # 避免除以零
    ndvi = (nir - red) / denominator
    return np.nan_to_num(ndvi, nan=0.0)  # 将 NaN 替换为 0.0

def find_Red_NIR_bands(listWavelength):
    R_wavelength = 682.5  # Red (625+740)/2
    NIR_wavelength = 850  # NIR

    rFound = nirFound = False
    rPreDifference = nirPreDifference = float('inf')
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

def load_and_process_hsi_data(hsi_data, wavelengths, k=5, max_iterations=10):
    # 执行 K-Means 聚类，并传入 logger
    m, c = kmeans(hsi_data, k, max_iterations)
    cluster_map = m.reshape(hsi_data.shape[:-1])  # 重塑为 2D 映射（行，列）

    # 找到红色和近红外波段
    red_band_index, nir_band_index = find_Red_NIR_bands(wavelengths)
    red_band = hsi_data[:, :, red_band_index].squeeze()
    nir_band = hsi_data[:, :, nir_band_index].squeeze()
    # print(f"Red band shape: {red_band.shape}")
    # print(f"NIR band shape: {nir_band.shape}")

    # 计算 NDVI
    ndvi = calculate_ndvi(nir_band, red_band)
    # print(f"NDVI shape after calculation: {ndvi.shape}")

    return cluster_map, ndvi


def build_mask_with_detections(hsi, detections_all):
    """
    Build a binary mask based on the outer rectangle of all detected Apriltags,
    and set the tag areas to 0.

    Args:
        hsi: (H, W, bands) hyperspectral data array.
        detections_all: list of detection dictionaries (each with a 'corners' key).

    Returns:
        mask: (H, W) binary mask where 1 indicates the retained region.
    """
    H, W, _ = hsi.shape
    # Concatenate corners from all detections
    all_corners = np.concatenate([d["corners"] for d in detections_all], axis=0)
    x_min_rect = int(np.min(all_corners[:, 0]))
    y_min_rect = int(np.min(all_corners[:, 1]))
    x_max_rect = int(np.max(all_corners[:, 0]))
    y_max_rect = int(np.max(all_corners[:, 1]))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y_min_rect:y_max_rect, x_min_rect:x_max_rect] = 1

    # Set the detected tag areas back to 0
    for d in detections_all:
        tag_x_min = int(np.min(d["corners"][:, 0]))
        tag_y_min = int(np.min(d["corners"][:, 1]))
        tag_x_max = int(np.max(d["corners"][:, 0]))
        tag_y_max = int(np.max(d["corners"][:, 1]))
        mask[tag_y_min:tag_y_max, tag_x_min:tag_x_max] = 0

    return mask

def apply_mask_to_hsi(hsi, mask):
    """
    Apply the binary mask to the hyperspectral cube. Pixels outside the mask
    are set to 0.

    Args:
        hsi: (H, W, bands) hyperspectral data array.
        mask: (H, W) binary mask.

    Returns:
        hsi_processed: a copy of the original HSI with excluded regions set to 0.
    """
    H, W, bands = hsi.shape
    hsi_processed = hsi.copy()
    mask_3d = np.repeat(mask[:, :, np.newaxis], bands, axis=2)
    hsi_processed[mask_3d == 0] = 0
    return hsi_processed

def detect_apriltags_multiscale(hsi, downscale_width=800):
    """
    Perform multi-scale Apriltag detection:
      1) Downsample the image (after a Gaussian blur) and run global detection.
      2) Group detections into four corners (top-left, top-right, bottom-left, bottom-right).
      3) For each corner, extract a dynamic ROI from the full-resolution blurred image,
         and run a local detection to get precise tag coordinates.

    Args:
        hsi: (H, W, bands) hyperspectral data array.
        downscale_width: target width for downsampling if the original width exceeds this value.

    Returns:
        final_detections: list of dictionaries with keys "tag_family", "corners", and "center".

    Raises:
        ValueError: if a corner tag is missing, multiple tags are detected in an ROI,
                    or if the detected tag is not of type "tag36h11".
    """
    H, W, bands = hsi.shape
    # Convert the first band to 8-bit grayscale and apply Gaussian blur to reduce noise
    img_gray_float = hsi[:, :, 0].astype(np.float32)
    img_gray = cv2.normalize(img_gray_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_gray_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Downsample if needed
    scale = downscale_width / float(W) if W > downscale_width else 1.0
    new_w = int(W * scale)
    new_h = int(H * scale)
    img_resized = cv2.resize(img_gray_blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create detector options without quad_sigma (since it's not supported)
    options = apriltag.DetectorOptions(
        quad_decimate=2.0,
        refine_edges=True
    )
    detector = apriltag.Detector(options)

    dets_resized = detector.detect(img_resized)
    if len(dets_resized) < 4:
        raise ValueError(f"Global detection on the downsampled image found {len(dets_resized)} tags; expected 4.")

    # Group detections into four corners based on center coordinates
    corners_map = {"tl": None, "tr": None, "bl": None, "br": None}
    half_w = new_w / 2.0
    half_h = new_h / 2.0
    for d in dets_resized:
        x, y = d.center
        if x < half_w and y < half_h:
            corners_map["tl"] = d
        elif x >= half_w and y < half_h:
            corners_map["tr"] = d
        elif x < half_w and y >= half_h:
            corners_map["bl"] = d
        else:
            corners_map["br"] = d
    for corner_key, detection in corners_map.items():
        if detection is None:
            raise ValueError(f"Downsampled image did not detect the {corner_key} corner Apriltag.")

    # Local detection for each corner on the full-resolution blurred image
    final_detections = []
    for corner_key, d in corners_map.items():
        # Convert detection coordinates from downsampled to original scale
        resized_corners = d.corners / scale
        x_min = int(np.min(resized_corners[:, 0])) - 50
        x_max = int(np.max(resized_corners[:, 0])) + 50
        y_min = int(np.min(resized_corners[:, 1])) - 50
        y_max = int(np.max(resized_corners[:, 1])) + 50

        # Clamp ROI coordinates within image bounds
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)

        roi_img = img_gray_blurred[y_min:y_max, x_min:x_max]
        # Optionally, apply an additional Gaussian blur to the ROI
        roi_img_blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)

        dets_roi = detector.detect(roi_img_blurred)
        if len(dets_roi) == 0:
            raise ValueError(f"No Apriltag detected in the local ROI of corner {corner_key}.")
        if len(dets_roi) > 1:
            raise ValueError(f"Multiple Apriltags detected in the local ROI of corner {corner_key}.")

        d2 = dets_roi[0]
        if d2.tag_family.decode() != "tag36h11":
            raise ValueError(f"The Apriltag in corner {corner_key} is not of type tag36h11.")

        # Convert ROI coordinates to global coordinates
        global_corners = d2.corners + np.array([x_min, y_min], dtype=np.float32)
        detection_info = {
            "tag_family": d2.tag_family.decode(),
            "corners": global_corners,
            "center": d2.center + np.array([x_min, y_min], dtype=np.float32)
        }
        final_detections.append(detection_info)

    return final_detections

def preprocess_hsi_with_apriltag_multiscale(hsi, file_path, downscale_width=800):
    """
    Preprocess the hyperspectral data by detecting four Apriltags (tag36h11) using
    a multi-scale approach, then construct a mask based on the outer rectangle of these
    tags while excluding the tag areas. Regions outside the mask are set to 0.

    Args:
        hsi: (H, W, bands) hyperspectral data array.
        file_path: path to the .hdr file.
        downscale_width: target width for downsampling during global detection.

    Returns:
        hsi_processed: hyperspectral data after applying the Apriltag-based mask.

    Raises:
        ValueError: if the file extension is not .hdr, if detection fails, or if tag types are incorrect.
    """
    # 1) File extension check (must be .hdr)
    if not file_path.lower().endswith('.bil'):
        raise ValueError("The file extension is not .bil. Please provide a valid .bil file.")

    # 2) Run the multi-scale Apriltag detection to get precise tag positions
    final_detections = detect_apriltags_multiscale(hsi, downscale_width=downscale_width)

    # 3) Build a mask using the detected tags
    mask = build_mask_with_detections(hsi, final_detections)

    # 4) Apply the mask to the HSI data (set excluded pixels to 0)
    hsi_processed = apply_mask_to_hsi(hsi, mask)
    return hsi_processed
