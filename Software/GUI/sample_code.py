from PyQt6.QtWidgets import QFileDialog,QMessageBox
import numpy as np

from spectral import *
import spectral.io.envi as envi
import os

import HSIHelper

def load_image(self):        
        imagePath, selectedFilter = QFileDialog.getOpenFileName(self, 'Open file', None,("Hyperspc Images(*.bil *.bip *.bsq)"))
        if imagePath == "":
            return
        
        self.image_path = imagePath                                                 
        rgb_image = None
        
        directory = os.path.dirname(self.image_path)
        base_name = os.path.basename(self.image_path)
        filename, extension = os.path.splitext(base_name)
        headerPath = directory + "/"+ filename + ".hdr"
        if (not os.path.exists(headerPath)): #if no header file, show error and return
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Header file not found!")
            msg_box.exec()
            return
        
        # check if it's PSI image format
        with open(headerPath, "r") as file:
            first_line = file.readline().strip()

        if first_line.startswith("BYTEORDER"): # PSI format
            dictMeta = HSIHelper.read_PSI_header(headerPath)
            headerPath = directory + "/"+ filename + "_envi.hdr" # new header file
            HSIHelper.create_envi_header(headerPath, dictMeta)
            

        self.hsi = envi.open(headerPath, self.image_path)
        tuple_rgb_bands = HSIHelper.find_RGB_bands([float(i) for i in self.hsi.metadata['wavelength']]) # metadata['wavelength'] is read as string; for CSIRO image, can use self.hsi.bands.centers
        rgb_image = get_rgb(self.hsi, tuple_rgb_bands) #(100, 54, 31)
        rgb_image = (rgb_image*255).astype(np.uint8)
        rgb_image = rgb_image.copy() # Spy don't load it to memory automatically, so must be copied

def save(self):
        # make a suggested name
        directory = os.path.dirname(self.image_path)
        base_name = os.path.basename(self.image_path)
        filename, extension = os.path.splitext(base_name)
        suggestedPath = directory + "/"+ filename + "_masked" + extension
        
        file_path,_ = QFileDialog.getSaveFileName(self, "Save Image", suggestedPath,("Hyperspc Images(*.bil *.bip *.bsq)"))
        if file_path:                 
            directory_save = os.path.dirname(file_path)
            base_name_save = os.path.basename(file_path)
            filename_save, extension_save = os.path.splitext(base_name_save)
            if extension_save == "":
                extension_save = self.hsi.metadata["interleave"]
            arr = self.hsi.load()
            arr[~self.viewer.mask_array] = [0] * self.hsi.nbands
            envi.save_image(directory_save + "/"+ filename_save + ".hdr", arr, force=True, ext=extension_save, interleave=self.hsi.metadata["interleave"], byteorder=self.hsi.byte_order, metadata=self.hsi.metadata)

