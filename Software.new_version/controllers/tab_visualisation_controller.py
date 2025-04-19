from io import BytesIO
import logging
import typing

from PIL.ImageQt import QPixmap
from matplotlib import pyplot as plt
import numpy as np
import spectral

from controllers.base_controller import BaseController
from widgets.tab_visualisation_view import TabVisualisationView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController

from Functions.Hypercube_Spectrum import Hypercube
from Functions.Visualization import Visualize_HSI


class TabVisualisationController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.modes = ["RGB", "NDVI", "EVI", "MCARI", "MTVI", "OSAVI", "PRI", "HyperCube"]
        self.tab_view = TabVisualisationView(self)

        self.mode_button_mapping = dict()
        self.mode_output = dict()
        for i, mode in enumerate(self.modes):
            self.mode_button_mapping[mode] = self.tab_view.radio_buttons[i]
            self.mode_output[mode] = None
        self.func = {
            "RGB": self.get_RGB,
            "NDVI": self.get_NDVI,
            "EVI": self.get_EVI,
            "MCARI": self.get_MCARI,
            "MTVI": self.get_MTVI,
            "OSAVI": self.get_OSAVI,
            "PRI": self.get_PRI,
            "HyperCube": self.get_hyper_cube,
        }

        # Connect radio buttons to slot
        for i in range(len(self.modes)):
            self.tab_view.radio_buttons[i].toggled.connect(self.on_click)

    def on_click(self):
        for i, mode in enumerate(self.modes):
            if self.tab_view.radio_buttons[i].isChecked():
                if self.mode_output[mode] is None:
                    self.mode_output[mode] = self.func[mode]()
                    self.tab_view.mode_view_mapping[mode].set_image(self.mode_output[mode])
                self.tab_view.stack.setCurrentIndex(i)
                break

    def on_load_file(self):
        for mode in self.mode_button_mapping:
            button = self.mode_button_mapping[mode]
            button.setEnabled(True)
            self.mode_output[mode] = None
        self.tab_view.radio_buttons[0].setChecked(True)
        self.on_click()

    @staticmethod
    def get_QPixmap(print_func) -> QPixmap:
        buf = BytesIO()
        # if im is plt:
        #     plt.savefig(buf, **kwargs)
        # else:
        #     plt.imsave(buf, im, **kwargs)
        print_func(buf)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        return pixmap

    def get_RGB(self):
        # From Functions.Visualization.Visualize_HSI.py - show_rgb
        tuple_rgb_bands = Visualize_HSI.find_RGB_bands(
            [float(i) for i in self.main_controller.hyperspectral_image.metadata["wavelength"]])  # metadata['wavelength'] is read as string; for CSIRO image, can use self.hsi.bands.centers
        rgb_image = spectral.get_rgb(self.main_controller.hyperspectral_image, tuple_rgb_bands)  # (100, 54, 31)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_image = rgb_image.copy()  # Spy don't load it to memory automatically, so must be copied
        self.logger.info(f"RGB Image Shape: {rgb_image.shape}")

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf, rgb_image))

    def get_NDVI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_ndvi
        ndvi_array = Visualize_HSI.calculate_ndvi(self.main_controller.hyperspectral_image)
        ndvi_image = (ndvi_array - np.min(ndvi_array)) / (np.max(ndvi_array) - np.min(ndvi_array))  # Normalize to 0-1

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf, ndvi_image, cmap="RdYlGn"))

    def get_EVI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_evi
        evi_array = Visualize_HSI.calculate_evi(self.main_controller.hyperspectral_image)

        # Replace NaNs and Infs with finite numbers (0)
        evi_array = np.nan_to_num(evi_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize EVI for visualization
        min_val = np.min(evi_array)
        max_val = np.max(evi_array)

        # Ensure the denominator is not zero
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-10

        evi_image = (evi_array - min_val) / range_val

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf, evi_image, cmap="RdYlGn"))

    def get_MCARI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_mcari
        mcari_array = Visualize_HSI.calculate_mcari(self.main_controller.hyperspectral_image)

        # Replace NaNs and Infs with finite numbers (0)
        mcari_array = np.nan_to_num(mcari_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize MCARI for visualization
        min_val = np.min(mcari_array)
        max_val = np.max(mcari_array)

        # Ensure the denominator is not zero
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-10

        mcari_image = (mcari_array - min_val) / range_val

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf, mcari_image, cmap="viridis"))

    def get_MTVI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_mtvi
        mtvi_array = Visualize_HSI.calculate_mtvi(self.main_controller.hyperspectral_image)

        # Replace NaNs and Infs with finite numbers (0)
        mtvi_array = np.nan_to_num(mtvi_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize MTVI for visualization
        min_val = np.min(mtvi_array)
        max_val = np.max(mtvi_array)

        # Ensure the denominator is not zero
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-10

        mtvi_image = (mtvi_array - min_val) / range_val

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf,mtvi_image, cmap="viridis"))

    def get_OSAVI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_osavi
        osavi_array = Visualize_HSI.calculate_osavi(self.main_controller.hyperspectral_image)
        osavi_image = (osavi_array - np.min(osavi_array)) / (np.max(osavi_array) - np.min(osavi_array))

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf,osavi_image, cmap="RdYlGn"))

    def get_PRI(self):
        # From Functions.Visualization.Visualize_HSI.py - show_pri
        pri_array = Visualize_HSI.calculate_pri(self.main_controller.hyperspectral_image)

        # Replace NaNs and Infs with finite numbers (0)
        pri_array = np.nan_to_num(pri_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize PRI for visualization
        min_val = np.min(pri_array)
        max_val = np.max(pri_array)

        # Ensure the denominator is not zero
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1e-10

        pri_image = (pri_array - min_val) / range_val

        return TabVisualisationController.get_QPixmap(lambda buf : plt.imsave(buf,pri_image, cmap="Spectral"))

    def get_hyper_cube(self):
        # From Functions.Hypercube_Spectrum.Hypercube.py - show_cube
        data = self.main_controller.hyperspectral_image
        assert len(data.shape) == 3
        self.logger.info(f"Hyperspectral image shape: {data.shape}")

        bands = list(Hypercube.find_RGB_bands([float(i) for i in data.metadata["wavelength"]]))
        r_band, g_band, b_band = bands[0], bands[1], bands[2]
        r_image = data[:, :, r_band]
        g_image = data[:, :, g_band]
        b_image = data[:, :, b_band]
        r_image[r_image < 0] = 0
        g_image[g_image < 0] = 0
        b_image[b_image < 0] = 0
        r_image_normalized = r_image / np.max(r_image)
        g_image_normalized = g_image / np.max(g_image)
        b_image_normalized = b_image / np.max(b_image)

        rgb_data = np.dstack((r_image_normalized, g_image_normalized, b_image_normalized))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        x_dim, y_dim, z_dim = data.shape
        X, Y = np.meshgrid(np.arange(y_dim), np.arange(x_dim))

        ax.plot_surface(X, Y, np.full((x_dim, y_dim), z_dim),
                        facecolors=rgb_data, rstride=1, cstride=1, shade=True)

        side1 = Hypercube.normalize_data(data[:, y_dim - 100, :])
        side2 = Hypercube.normalize_data(data[x_dim - 100, :, :])

        X_side1, Z_side1 = np.meshgrid(np.arange(x_dim), np.arange(z_dim))
        Y_side2, Z_side2 = np.meshgrid(np.arange(y_dim), np.arange(z_dim))

        side1 = np.squeeze(side1).T
        side2 = np.squeeze(side2).T

        ax.plot_surface(np.full_like(X_side1, y_dim), X_side1, Z_side1,
                        facecolors=plt.cm.viridis(side1), rstride=1, cstride=1, shade=True)
        ax.plot_surface(Y_side2, np.full_like(Y_side2, x_dim), Z_side2,
                        facecolors=plt.cm.viridis(side2), rstride=1, cstride=1, shade=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Wavelength')
        ax.set_xlim(0, y_dim)
        ax.set_ylim(0, x_dim)
        ax.set_zlim(0, z_dim)

        ax.view_init(elev=30, azim=45)
        plt.title('Hypercube Visualization')

        return TabVisualisationController.get_QPixmap(lambda buf : plt.savefig(buf, dpi=300, bbox_inches="tight"))
