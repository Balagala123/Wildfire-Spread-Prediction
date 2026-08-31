# Wildfire Spread Prediction Using Deep Learning

### DCT-Based U-Net Approach for Pixel-Level Wildfire Spread Prediction

## Overview

This project explores deep learning approaches for predicting wildfire spread using multi-source satellite remote sensing data.

The primary HadamardU-Net model combines Discrete Cosine Transform (DCT) operations, learnable thresholding, dropout, and residual connections for pixel-level wildfire `FireMask` prediction.

## Problem

Wildfire spread prediction is challenging due to complex spatial patterns, environmental variability, class imbalance, and uncertain wildfire boundaries.

The objective is to predict wildfire regions from multi-source satellite remote sensing data using deep learning.

## Approach

The overall pipeline is:

**Satellite Remote Sensing Data → Preprocessing → HadamardU-Net → FireMask Prediction → Evaluation**

### Data Processing

* Multi-source satellite remote sensing data
* TFRecord-based data pipeline
* Data preprocessing using Google Earth Engine
* Random flipping and rotation for data augmentation
* 64 × 64 image samples
* Binary FireMask prediction

### Model Architecture

The HadamardU-Net is a U-Net-inspired encoder-decoder architecture with frequency-domain processing.

Key components:

* Convolutional encoder and decoder layers
* Discrete Cosine Transform (DCT)
* Learnable frequency-domain parameters
* Learnable soft-thresholding
* Batch normalization
* Dropout
* Skip connections
* Residual connections

## Loss Function

The training objective combines three loss functions:

* Weighted Binary Cross-Entropy
* Dice Loss
* Focal Loss
  
```
Combined Loss =
0.4 × Weighted BCE +
0.3 × Dice Loss +
0.3 × Focal Loss
```

This combination helps address class imbalance and improve wildfire-region prediction.

## Evaluation

Model performance is evaluated using:

* **AUC**
* **Precision**
* **Recall**
* **Intersection over Union (IoU)**
* **F1-score**

The project also includes probability threshold optimization for converting model probabilities into binary wildfire predictions.
Training and validation loss curves and wildfire prediction visualizations are generated as part of the evaluation process.


## Technologies

* Python
* PyTorch
* TensorFlow
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib
* OpenCV
* Google Earth Engine

## Data

The project uses multi-source satellite remote sensing data stored in **TFRecord** format. Dataset files should be obtained separately and the training, evaluation, and test paths configured locally.
