# TechLauncher-HSISR
This is the code repository for the TechLauncher project Hyperspectral Image Super-resolution.

The project's client comes from the Australian Plant Phenomics Facility (APPF). At present, APPF has three hyperspectral cameras from different brands, each with its own unique working wavelength range. The project aims to increase the spatial resolution of the images captured by the HSI cameras by a factor of 2 or more. The resulting super-resolution hyperspectral images will facilitate more precise subsequent computer vision tasks, including object segmentation, classification, detection, and tissue or component analysis. The primary measure of success will be the improvement in spatial resolution while maintaining spectral fidelity.

The project's scope is shown in the below image. The original pipeline is:
1. get the raw hyperspectral images(HSIs) from the camera.
2. do some data preprocessing, for example, collaboration, denoising;
3. utilize the preprocessed HSIs to conduct some computer vision tasks, including segmentation, detection, etc.
4. get metrics from the CV applications. 

![pipeline and scope](https://github.com/squashking/TechLauncher-HSISR/blob/main/pipeline.png)

This leaf project focuses on the progress between preprocessing and CV application, as the low spatial resolution of HSIs does influence the performance of following CV applications, we aim to add a new factor between preprocessing and CV application to enhance the spatial resolution by a factor of 2 or more to capture finer details of plants.
