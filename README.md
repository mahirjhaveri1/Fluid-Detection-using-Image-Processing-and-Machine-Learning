# Fluid Detection Using Image Processing and Machine Learning

## Overview
This project presents an embedded vision system designed to detect fluid levels in bottles and classify them based on a predefined threshold (75% fill level). The system is implemented on a Raspberry Pi and integrates image processing techniques with a Convolutional Neural Network (CNN) for accurate classification.

The solution captures images using a camera module, processes them to extract relevant features, and classifies each bottle as acceptable or rejected based on its fluid level.

---

## Features
- Automated fluid level detection using image processing
- CNN-based classification for accurate decision making
- Custom dataset creation and preprocessing
- Real-time inference on Raspberry Pi
- Threshold-based validation (75% fill level)
- Lightweight and edge-deployable system

---

## System Architecture
1. Image Acquisition  
   - Images are captured using a Raspberry Pi camera module  

2. Preprocessing  
   - Image resizing, normalization, and segmentation  

3. Feature Extraction  
   - Relevant visual patterns are extracted  

4. Classification  
   - CNN model classifies bottle fill level (e.g., 0%, 25%, 50%, 75%, 100%)  

5. Decision Output  
   - Bottles are marked as accepted or rejected based on the 75% threshold  

---

## Tech Stack
- Python  
- OpenCV  
- TensorFlow / Keras  
- Raspberry Pi (3B)  

---

## Dataset
A custom dataset was created for training and validation. It consists of labeled images categorized based on fluid levels:
- 0% (Empty)
- 25%
- 50%
- 75% (Target level)
- 100%
Dataset structure:
