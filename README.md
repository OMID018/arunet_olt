[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMID018/arunet_olt/blob/main/test_colab.ipynb)

# ARUNet_OLT – Old Logging Trail Segmentation Using Deep Learning
This repository contains a trained ARUNet model and a complete inference pipeline for segmenting Old Logging Trails (OLTs) in boreal forests using high-resolution LiDAR-derived raster layers. The model was trained on 256×256 input patches of stacked CHM, DSM, and DEM data, with a cell size of 0.4 meters. The output is an image that includes the segmented logging trails, polygons of OLTs and the centerlines.

---

## 🧠 Model Description

- **Architecture:** Attention Residual U-Net (ARUNet) for segmenting of OLTs
- <img width="1280" height="720" alt="RAUNET3" src="https://github.com/user-attachments/assets/4c9a9774-4164-4c7b-b3ce-cf60a78575e9" />
- **Input:** 256×256 patches with 3 channels (CHM, DSM, DEM)
- **Output:** Predicted probability map of OLTs, segmented raster of OLts, polygons of OLTs, and centerlines of OLTs
- **Resolution:** 0.4 meters
- **Framework:** TensorFlow 2.x, ArcGIS api

---

## 📂 Repository Structure

## Installation
STEP 1: Clone the repository

!git clone https://github.com/OMID018/arunet_olt.git
%cd arunet_olt

STEP 2: Install requirements
!pip install -r requirements.txt

STEP 3: Use input data to perform inference with the model

STEP 4: Run the full pipeline
!python main.py


