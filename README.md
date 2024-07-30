# NeuroScan: Machine Learning Application for Brain Tumor Detection 

This project implements an image classification system using Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. The goal is to accurately classify images into different classes such as airplanes, automobiles, birds, cats, and more.
We utilize TensorFlow and Keras, popular deep learning libraries, to design, train, and evaluate a CNN model for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. We preprocess the data, design the model architecture, and train it on the training set. Finally, we evaluate the model's performance on the test set.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
4. [Image Processing](#image-processing)
   - [Thresholding](#thresholding)
   - [Erosion](#erosion)
   - [Dilation](#dilation)
5. [Model Performance](#model-performance)
   - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
   - [Naive Bayes Classifier](#naive-bayes-classifier)
10. [Research Mentors](#research-mentors)
11. [Citations](#citations)

## Overview
In this project, we utilize TensorFlow and Keras, popular deep learning libraries, to design, train, and evaluate a CNN model for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. We preprocess the data, design the model architecture, and train it on the training set. Finally, we evaluate the model's performance on the test set.

## Installation
To run this project, you need to have Python 3.11 or below installed along with the following dependencies:

 **Install the required libraries:**
    Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    ```

    Activate the virtual environment:
    - On Windows:
      ```bash
      env\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source env/bin/activate
      ```

    Install the libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
## Image Processing

### Binary Thresholding

Binary thresholding is applied to the grayscale image using the `cv2.threshold` function. The operation is performed as follows:

```python
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
```
Explanation:

    Source Image (gray): This is the grayscale image that will be thresholded.
    Threshold Value (45): Pixels with intensity values below this threshold are set to 0. Pixels with intensity values equal to or above this threshold are set to the maximum value (255).
    Maximum Value (255): The value assigned to pixels that meet or exceed the threshold.
    Thresholding Type (cv2.THRESH_BINARY): Specifies that a binary thresholding operation is to be performed. Pixels below the threshold become 0, and those above or equal to the threshold become 255.

The cv2.threshold function returns a tuple, where the second element ([1]) is the thresholded image stored in thresh.

## Erosion

Erosion is applied to the thresholded image to reduce noise and small details. This is accomplished using the cv2.erode function:

python
```
thresh = cv2.erode(thresh, None, iterations=2)
```
Explanation:

    Source Image (thresh): The image that has undergone binary thresholding and is now subjected to erosion.
    Kernel (None): Uses a default 3x3 rectangular kernel for the erosion operation.
    Iterations (2): Specifies that the erosion operation should be applied twice.

Erosion reduces the size of foreground objects by eroding away their boundaries, which helps in removing small noise or details from the image.

## Dilation

Following erosion, dilation is performed to enlarge the boundaries of the foreground objects and fill in gaps. This is done using the cv2.dilate function:

python
```
thresh = cv2.dilate(thresh, None, iterations=2)
```
Explanation:

    Source Image (thresh): The image that has undergone erosion and is now subjected to dilation.
    Kernel (None): Uses a default 3x3 rectangular kernel for the dilation operation.
    Iterations (2): Specifies that the dilation operation should be applied twice.

Dilation expands the boundaries of foreground objects, which helps in closing small holes and connecting nearby objects. This process makes the objects in the image larger and more continuous.

## Purpose of These Operations

These image processing techniques—thresholding, erosion, and dilation—are used to preprocess the image by separating objects from the background and extracting specific regions of interest based on intensity values. By combining these operations, we enhance the quality of the image for further analysis or object detection.

## Model Performance

### Convolutional Neural Network (CNN)

The CNN model was evaluated on the CIFAR-10 dataset with the following results:

- **Test Accuracy**: 0.8646
- **Test Loss**: 0.3562

#### Training Details

The CNN model was trained over 30 epochs with the following progress:

| Epoch | Accuracy | Loss  | Validation Accuracy | Validation Loss |
|-------|----------|-------|---------------------|-----------------|
| 1     | 0.7277   | 0.6232| 0.7810              | 0.5152          |
| 2     | 0.8118   | 0.4371| 0.7954              | 0.4687          |
| 3     | 0.8448   | 0.3682| 0.8012              | 0.4696          |
| 4     | 0.8396   | 0.3886| 0.8357              | 0.4119          |
| 5     | 0.8667   | 0.3237| 0.7954              | 0.4377          |
| ...   | ...      | ...   | ...                 | ...             |
| 20    | 0.9249   | 0.1847| 0.8530              | 0.3790          |

The CNN model showed consistent improvement in accuracy and reduction in loss throughout the training epochs. It achieved a high validation accuracy of approximately 86.5%, demonstrating strong performance in classifying images from the CIFAR-10 dataset.

### Support Vector Machine (SVM)

The SVM classifier was evaluated with the following performance metrics:

- **Accuracy**: 0.8156

#### Classification Report

The classification report provides detailed performance metrics for each class:

| Class                    | Precision | Recall | F1-Score | Support |
|--------------------------|-----------|--------|----------|--------|
| No Brain Tumor Detected  | 0.46      | 0.72   | 0.56     | 107    |
| Brain Tumor Detected     | 0.83      | 0.62   | 0.71     | 240    |
| **Accuracy**             |           |        | 0.65     | 347    |
| **Macro Average**        | 0.64      | 0.67   | 0.63     | 347    |
| **Weighted Average**     | 0.72      | 0.65   | 0.66     | 347    |

The SVM classifier demonstrated a good overall accuracy of 81.56%, with varying precision and recall across different classes. The F1-score for each class provides insight into the balance between precision and recall.

### Naive Bayes Classifier

The Naive Bayes classifier was evaluated with the following results:

- **Accuracy**: 0.6484
- **Number of Examples**: 4625
- **Number of Positive Examples**: 3253
- **Number of Negative Examples**: 1372

#### Classification Report

| Metric                    | Value    |
|---------------------------|----------|
| **Accuracy**              | 0.6484   |
| **Number of Examples**    | 4625     |
| **Number of Positive Examples** | 3253 |
| **Number of Negative Examples** | 1372 |

The Naive Bayes classifier achieved an accuracy of approximately 64.84%. This model was tested on a dataset with a significant number of positive examples, showing moderate performance compared to the SVM and CNN models.

### Summary

- The **CNN model** outperformed both the SVM and Naive Bayes classifiers, with the highest accuracy and lowest loss, demonstrating its effectiveness in handling complex image data.
- The **SVM classifier** showed competitive performance with good precision and recall, particularly for detecting brain tumors.
- The **Naive Bayes classifier** provided a baseline performance with moderate accuracy and was less effective compared to the other models.

These results illustrate the strengths and limitations of each classification approach and help in understanding their applicability to different types of image classification tasks.

## Research Mentors
This project was guided by Research Mentors Dr. Mokhtar Sadok, Dr. Indranil Mukhopadhyay, and Dr. Mohammad Akram Hossain during my internship as a Data Science and Machine Learning Intern at Ascend Technology Inc.

## Citations
Sadok, M. (2021, August 8). Artificial Intelligence: A paradigm shift in the pharmaceutical industry - use case of cancer detection. Digitale Transformation - jetzt die Chancen aktiv nutzen! https://www.strategy-transformation.com/artificial-intelligence-a-paradigm-shift-in-the-pharmaceutical-industry-use-case-of-cancer-detection/ 


