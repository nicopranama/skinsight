# SkinSight – Skin Type Classification using ResNet50

**SkinSight** is a deep learning project that uses **ResNet50** to classify skin types from images.  
This project leverages **transfer learning** to achieve high accuracy in predicting skin categories such as **Oily**, **Dry**, **Normal**, and **Combination** skin.

![Accuracy](https://img.shields.io/badge/accuracy-83%25-brightgreen)

---

## Project Overview

SkinSight uses a pre-trained ResNet50 model, fine-tuned on a custom dataset of skin images, to classify skin types with an achieved accuracy of **83%**. The model supports easy input from images and processes them into predictable skin categories.

---

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV (cv2)
- Scikit-learn

---

## How to Run

- Open Jupyter Notebook or Google Colab
- Load and run all cells in SkinSight.ipynb
- Upload a skin image when prompted (or modify code to batch-load)
- View predicted skin type in the output cell

---

## Model Notes

- Base Model: ResNet50 (include_top=False)
- Image size: resized to 224x224
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Achieved Accuracy: 83% on test dataset
- Image preprocessing with OpenCV and NumPy

---

## Future Improvements

- Improve dataset balance and class variety
- Integrate Grad-CAM for interpretability
- Deploy as an API or web application
- Add real-time prediction UI for mobile/web
