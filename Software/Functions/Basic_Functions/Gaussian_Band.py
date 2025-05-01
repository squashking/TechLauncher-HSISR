import numpy as np
 import matplotlib
 
 matplotlib.use('TkAgg')
 import matplotlib.pyplot as plt
 from scipy.stats import norm
 
 
 def read_wavelengths_from_hdr(hdr_path):
     wavelengths = []
     with open(hdr_path, 'r') as file:
         in_block = False
         for line in file:
             if 'WAVELENGTHS' in line:
                 in_block = True
                 continue
             if 'WAVELENGTHS_END' in line:
                 break
             if in_block:
                 try:
                     wavelengths.append(float(line.strip()))
                 except ValueError:
                     continue
     return np.array(wavelengths)
 
 
 def read_bil_data(bil_path, rows, cols, bands, dtype=np.uint16):
     raw = np.fromfile(bil_path, dtype=dtype)
     cube = raw.reshape((rows, bands, cols))
     return np.transpose(cube, (0, 2, 1))  # → (rows, cols, bands)
 
 
 def gaussian_weighted_rgb(hsi_cube, wavelengths, sigma=20, r_center=650, g_center=550, b_center=450):
     def weights(center): return norm.pdf(wavelengths, center, sigma) / np.sum(norm.pdf(wavelengths, center, sigma))
 
     R = np.tensordot(hsi_cube, weights(r_center), axes=([2], [0]))
     G = np.tensordot(hsi_cube, weights(g_center), axes=([2], [0]))
     B = np.tensordot(hsi_cube, weights(b_center), axes=([2], [0]))
     rgb = np.stack([R, G, B], axis=-1)
     return (rgb - rgb.min()) / (rgb.max() - rgb.min())
 
 
 hdr_path = "2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.hdr"
 bil_path = "2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.bil"
 nrows, ncols, nbands = 385, 500, 480
 
 wavelengths = read_wavelengths_from_hdr(hdr_path)
 cube = read_bil_data(bil_path, nrows, ncols, nbands)
 
 sigma_values = [5, 10, 20, 40, 80]
 #sigma_values = [20]
 
 plt.figure(figsize=(4 * len(sigma_values), 4))
 for i, sigma in enumerate(sigma_values):
     rgb = gaussian_weighted_rgb(cube, wavelengths, sigma=sigma)
 
 
     plt.subplot(1, len(sigma_values), i + 1)
     plt.imshow(rgb)
     plt.title(f"σ = {sigma}")
     plt.axis("off")
 
 
 def mean_rgb_from_ranges(hsi_cube, wavelengths, r_range=(640, 700), g_range=(520, 580), b_range=(450, 500)):
     def band_range(wrange):
         return np.where((wavelengths >= wrange[0]) & (wavelengths <= wrange[1]))[0]
 
     r_idx = band_range(r_range)
     g_idx = band_range(g_range)
     b_idx = band_range(b_range)
 
     R = np.mean(hsi_cube[:, :, r_idx], axis=2)
     G = np.mean(hsi_cube[:, :, g_idx], axis=2)
     B = np.mean(hsi_cube[:, :, b_idx], axis=2)
 
     rgb = np.stack([R, G, B], axis=-1)
     return (rgb - rgb.min()) / (rgb.max() - rgb.min())
 
 
 rgb_mean = mean_rgb_from_ranges(cube, wavelengths)
 
 plt.figure(figsize=(8, 8))
 plt.imshow(rgb_mean)
 plt.title("Mean RGB from Band Ranges (no Gaussian)")
 plt.axis("off")
 plt.show()
 
 plt.suptitle("Gaussian Weighted RGB Comparison for Different σ", fontsize=16)
 plt.tight_layout()
 plt.show()
