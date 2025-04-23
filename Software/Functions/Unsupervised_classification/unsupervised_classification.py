# unsupervised_classification.py

import numpy as np
from spectral import kmeans
from pyapriltags import Detector
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
    # Ensure we have a NumPy array, not a BilFile or EnviFile
    if not isinstance(hsi_data, np.ndarray):
        hsi_data = hsi_data[:, :, :]  # slice will load the full cube as ndarray

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

def detect_apriltags(hsi):
    """
    Try several visible-light bands and detect exactly four AprilTags.
    Returns a list of detections with keys 'tag_family','corners','center'.
    """
    H, W, B = hsi.shape

    # Candidate band indices (25%, 50%, 75% through the cube)
    candidates = [B // 4, B // 2, (3 * B) // 4]
    detector = Detector(quad_decimate=1.0, refine_edges=True)

    dets = None
    gray_full = None

    # 1) Find a band that yields at least 4 tags
    for vis_idx in candidates:
        band = hsi[:, :, vis_idx].astype(np.float32)
        gray = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        det_list = detector.detect(gray)
        if len(det_list) >= 4:
            dets = det_list
            gray_full = gray
            print(f"Using band {vis_idx} for AprilTag detection, found {len(dets)} tags.")
            break

    if dets is None:
        raise ValueError(f"No AprilTags found in bands {candidates}.")

    # 2) Partition into four corners by tag center
    half_w, half_h = W / 2.0, H / 2.0
    corners_map = {"tl": None, "tr": None, "bl": None, "br": None}
    for d in dets:
        x, y = d.center
        if   x < half_w and y < half_h: corners_map["tl"] = d
        elif x >= half_w and y < half_h: corners_map["tr"] = d
        elif x < half_w and y >= half_h: corners_map["bl"] = d
        else:                            corners_map["br"] = d

    for key, val in corners_map.items():
        if val is None:
            raise ValueError(f"Missing '{key}' corner AprilTag.")

    # 3) Local refine each corner (no downsampling)
    final_detections = []
    for corner_key, d in corners_map.items():
        xs = d.corners[:, 0]
        ys = d.corners[:, 1]
        x0 = max(0, int(xs.min()) - 50)
        x1 = min(W, int(xs.max()) + 50)
        y0 = max(0, int(ys.min()) - 50)
        y1 = min(H, int(ys.max()) + 50)

        roi = gray_full[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        dets_roi = detector.detect(roi)

        if len(dets_roi) != 1:
            raise ValueError(
                f"Corner '{corner_key}' local detection found {len(dets_roi)} tags; expected exactly 1."
            )
        d2 = dets_roi[0]
        if d2.tag_family.decode() != "tag36h11":
            raise ValueError(
                f"Corner '{corner_key}' wrong tag family: {d2.tag_family.decode()}."
            )

        # Map local ROI coords back to global image coords
        global_corners = d2.corners + np.array([x0, y0], dtype=np.float32)
        final_detections.append({
            "tag_family": d2.tag_family.decode(),
            "corners":    global_corners,
            "center":     d2.center + np.array([x0, y0], dtype=np.float32)
        })

    return final_detections


def preprocess_hsi_with_apriltag_multiscale(hsi, file_path, downscale_width=800):
    """
    1) Ensure hsi is a NumPy ndarray, not a BilFile.
    2) Verify .bil extension.
    3) Detect exactly 4 Apriltags (no downsampling).
    4) Build & apply mask to zero-out everything outside the tags’ outer rectangle.
    5) Crop the masked HSI to the minimal bounding rectangle enclosing all Apriltags.
    """
    # --- 1) Force ndarray conversion ---
    if not isinstance(hsi, np.ndarray):
        # slice will load the full cube into memory as an ndarray
        hsi = hsi[:, :, :]

    H, W, _ = hsi.shape

    # --- 2) Extension check ---
    if not file_path.lower().endswith('.bil'):
        raise ValueError("The file extension is not .bil. Please provide a valid .bil file.")

    # --- 3) Apriltag detection ---
    final_detections = detect_apriltags(hsi)
    # (Assumes detect_apriltags returns a list of 4 dicts with keys "corners" and "center")

    # --- 4) Build & apply mask ---
    mask = build_mask_with_detections(hsi, final_detections)
    hsi_masked = apply_mask_to_hsi(hsi, mask)

    # --- 5) Compute bounding box over all tag corners ---
    all_corners = np.vstack([d["corners"] for d in final_detections])
    x_min = max(0, int(np.floor(all_corners[:, 0].min())))
    x_max = min(W, int(np.ceil (all_corners[:, 0].max())))
    y_min = max(0, int(np.floor(all_corners[:, 1].min())))
    y_max = min(H, int(np.ceil (all_corners[:, 1].max())))

    # Crop the masked HSI cube to that rectangle
    hsi_cropped = hsi_masked[y_min:y_max, x_min:x_max, :]

    return hsi_cropped

