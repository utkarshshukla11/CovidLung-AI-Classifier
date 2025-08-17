# 🩺 Covid-19 & Pneumonia X-Ray Detector

Welcome to the **Covid-19 & Pneumonia X-Ray Detector** project on GitHub! 🌡️📊  
This is a deep learning project I’ve been working on to help doctors interpret chest X-rays.  
The model can detect and classify **four conditions**:  

- ✅ Normal (Healthy)  
- 🦠 Covid-19  
- 🧬 Viral Pneumonia  
- 🧫 Bacterial Pneumonia  

Pretty cool, right? 🚀  

---

## 🌟 What’s This All About?

Doctors sometimes struggle to tell the difference between **Covid-19, Viral Pneumonia, and Bacterial Pneumonia** just by looking at X-rays.  
A wrong call or delay can put lives at risk.  

This project builds an AI model that:  
- 📌 Classifies X-rays into 4 categories with strong accuracy  
- 📌 Helps doctors make quicker, more confident diagnoses  
- 📌 Lowers the chances of misdiagnosis  
- 📌 Automates detection to speed things up in hospitals  

---

## 📊 Dataset


Each category (Normal, Covid-19, Viral Pneumonia, Bacterial Pneumonia) contains **133 X-ray images** → ~532 images in total.

---

## 🎉 What Can It Do?

- 🧠 **Smart Classification** → Detects if an X-ray is Normal, Covid-19, Viral, or Bacterial Pneumonia.  
- ⚡ **Powerful Tech** → Uses **ResNet50**, a top-notch deep learning model for image recognition.  
- 🎨 **Image Boosting** → Applies data augmentation to make the model more robust.  
- 📈 **Cool Visuals** → Accuracy/Loss graphs & confusion matrix to track performance.  
- 💾 **Best Model Saved** → Keeps the best version based on validation performance.  

---

## ⚙️ Tech Stack

- Python 🐍  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib, Seaborn  

---

## 🚀 Code Setup

Here’s the core setup to get started:  

```python
import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


---

## 👤 Author
**Utkarsh Shukla**  
🔗 [LinkedIn](https://www.linkedin.com/in/utkarshshukla111)  

---
