# Transformer-Based Diabetic Retinopathy Severity Grading Using Enhanced Fundus Images

## Overview
This project focuses on classifying retinal fundus images into five diabetic retinopathy severity levels using transformer-based deep learning architectures. It includes image enhancement, model evaluation, and an interactive Streamlit application for severity prediction.

## Dataset
- **Dataset:** [Diabetic Retinopathy Fundus Images – Kaggle](https://www.kaggle.com/datasets/dokkuvenkatalakshmi/diabetic-retinopathy-fundus-images)
- **Original Images:** 3,662
- **Images After Duplicate Removal:** 3,534
- **Classes:** No_DR, Mild, Moderate, Severe, and Proliferate_DR.
- **Image Resolution:** 224 × 224 × 3

## Project Workflow
1. **Dataset Analysis:** Perform statistical analysis and EDA to examine class distribution, image dimensions, RGB intensities, brightness, and sharpness.
2. **Preprocessing:** Apply image validation, duplicate removal, CLAHE contrast enhancement, Non-Local Means denoising, resizing, and normalization.
3. **Data Augmentation:** Apply augmentation to training images while keeping validation and testing data free from random augmentation.
4. **Dataset Splitting:** Use stratified splitting with 70% training, 15% validation, and 15% testing.
5. **Model Development:** Implement and evaluate:
   - Vision Transformer (ViT)
   - Swin Transformer
   - CNN-ViT Hybrid
6. **Model Evaluation:** Analyse Accuracy, Precision, Recall, F1-Score, and Cohen's Kappa using classification reports, learning curves, confusion matrices, and ROC curves.
7. **Deployment:** Develop a Streamlit application for diabetic retinopathy severity prediction.

## Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| ViT | 80.60% | 81.20% |
| Swin Transformer | 78.53% | 65.33% |
| CNN-ViT Hybrid | 75.33% | 75.37% |

## Technologies Used
Python, TensorFlow, Keras, KerasHub, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit, and Pillow.

## Project Structure
```text
Diabetic-Retinopathy/
│
├── Code.ipynb
├── Results/
│   ├── EDA/
│   ├── Model Evaluation/
│   └── Application Screenshots/
│
└── App/
    ├── app.py
    ├── requirements.txt
    └── assets/
        └── retina_background.jpg
```

## Objective
To develop and evaluate transformer-based deep learning models for diabetic retinopathy severity grading and provide an interactive retinal image classification system for research and educational purposes.
