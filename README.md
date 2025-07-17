# ARUNet_OLT – Old Logging Trail Segmentation Using Deep Learning

This repository contains a trained ARUNet model and a complete inference pipeline for segmenting Old Logging Trails (OLTs) in boreal forests using high-resolution LiDAR-derived raster layers. The model was trained on 256×256 input patches of stacked CHM, DSM, and DEM data.

---

## 🧠 Model Description

- **Architecture:** Attention Residual U-Net (ARUNet)
- **Input:** 256×256 patches with 3 channels (CHM, DSM, DEM)
- **Output:** Predicted probability map of OLTs
- **Resolution:** 0.4 meters
- **Framework:** TensorFlow 2.x

---

## 📂 Repository Structure

