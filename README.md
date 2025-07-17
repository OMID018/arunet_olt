[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMID018/arunet_olt/blob/main/test_colab.ipynb)

# ARUNet_OLT – Old Logging Trail Segmentation Using Deep Learning
This repository contains a trained ARUNet model and a complete inference pipeline for segmenting Old Logging Trails (OLTs) in boreal forests using high-resolution LiDAR-derived raster layers. The model was trained on 256×256 input patches of stacked CHM, DSM, and DEM data, with a cell size of 0.4 meters. The output is an image that includes the segmented logging trails, polygons of OLTs and the centerlines.

---

## 🧠 Model Description

- **Architecture:** Attention Residual U-Net (ARUNet) for segmenting of OLTs
- <img width="1280" height="720" alt="RAUNET2" src="https://github.com/user-attachments/assets/55b0f2b8-4da1-4533-8b44-09ae8bfe9a14" />
- **Input:** 256×256 patches with 3 channels (CHM, DSM, DEM)
- **Output:** Predicted probability map of OLTs
- **Resolution:** 0.4 meters
- **Framework:** TensorFlow 2.x

---

## 📂 Repository Structure

