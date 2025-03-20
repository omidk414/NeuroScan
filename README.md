# NeuroScan: AI-Powered Brain Tumor Detection

## Problem & Motivation

Brain tumors are a significant contributor to global morbidity and mortality. Early detection is crucial for improving prognosis, but diagnosing brain tumors through manual analysis of MRI scans is time-consuming, requires significant expertise, and is prone to human error. As the volume of medical imaging data grows, healthcare professionals face challenges in providing timely and consistent diagnoses.

NeuroScan addresses this challenge by leveraging advanced deep learning techniques to automatically detect and classify brain tumors from MRI scans, ensuring faster and more reliable diagnoses while improving the efficiency of healthcare systems.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources & Methodology](#data-sources--methodology)
3. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Dataset Preparation](#dataset-preparation)
   - [Running the Application](#running-the-application)
4. [System Architecture](#system-architecture)
5. [Image Preprocessing Techniques](#image-preprocessing-techniques)
   - [Binary Thresholding](#binary-thresholding)
   - [Erosion](#erosion)
   - [Dilation](#dilation)
6. [Model Evaluation & Results](#model-evaluation--results)
   - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
   - [Naive Bayes Classifier](#naive-bayes-classifier)
7. [Key Learnings & Impact](#key-learnings--impact)
8. [Future Development Roadmap](#future-development-roadmap)
9. [Acknowledgements](#acknowledgements)
10. [References & Resources](#references--resources)
11. [Disclaimer](#disclaimer)

---

## Project Overview

NeuroScan utilizes cutting-edge deep learning methods to automate the detection and classification of brain tumors from MRI images. Our mission is to empower medical professionals with a reliable tool that enhances diagnostic speed, accuracy, and consistency.

---

## Data Sources & Methodology

NeuroScan leverages publicly available medical imaging datasets such as the **Brain Tumor Dataset**, which includes MRI scans labeled as healthy or tumor-positive. The dataset undergoes preprocessing using techniques like thresholding, erosion, and dilation to improve image quality and enhance model accuracy.

Our supervised learning approach employs **Convolutional Neural Networks (CNNs)** to capture spatial features critical for accurate tumor classification.

---

## Getting Started
### Installation
To run this project, you need to have Python 3.11 or below installed along with the following dependencies:

**Install the required libraries:**
Create a virtual environment (optional but recommended):
    ```
    python -m venv env
    ```

Activate the virtual environment:
- On Windows:
      ```
      env\Scripts\activate
      ```
- On macOS/Linux:
      ```
      source env/bin/activate
      ```

Install the libraries from `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```

---

## System Architecture

NeuroScan's architecture employs CNNs trained on preprocessed MRI images stored in a structured dataset pipeline. It includes modules for preprocessing, model training (CNN, SVM, Naive Bayes), prediction, and output generation for diagnostic results.

---
    
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

## Model Evaluation 

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

---

## Key Learnings & Impact

This project demonstrated the importance of effective image preprocessing and parameter tuning in deep learning applications for medical diagnostics. NeuroScan’s flexibility allows scalability into clinical environments, significantly enhancing diagnostic workflows.

---

## Future Development Roadmap

Future enhancements include:

- **Dataset Expansion**: Integrating diverse MRI data to improve generalizability.
- **Advanced Models**: Exploring transfer learning and hybrid architectures.
- **Real-Time Integration**: Developing real-time processing capabilities for clinical use.

---

## Acknowledgements

Special thanks to:

- **Dr. Mokhtar Sadok**, for insights into neural network optimization for medical imaging.
- **Dr. Indranil Mukhopadhyay**, for guidance on advanced data preprocessing techniques.
- **Dr. Mohammad Akram Hossain**, for expertise in deep learning architectures applied to healthcare.
- Friends, family members, colleagues who participated as beta testers and provided valuable feedback during development.
- Authors of key research articles listed below that informed our approach.

---

## References & Resources

1. ["Brain Tumor Detection Using CNN"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11353951/)
2. ["Efficient Brain Tumor Classification Framework"](https://www.aimspress.com/article/doi/10.3934/mbe.2023528)
3. ["MRI-Based Brain Tumor Diagnosis with Transfer Learning"](https://www.mdpi.com/2076-3417/10/6/1999)
4. ["Automated Detection of Brain Tumors Using CNNs"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10238087/)
5. ["Brain Tumor Classification Using Pixel Distribution"](https://www.nature.com/articles/s41598-020-59055-x)

---

## Disclaimer

The NeuroScan application is designed solely for educational purposes and research exploration into AI-based medical diagnostics. It should not replace professional medical advice or diagnosis.

Users must consult healthcare professionals before making any medical decisions based on this tool's outputs.

The authors disclaim liability related to accuracy or consequences arising from the use of NeuroScan results.

By using this application, users acknowledge these terms and agree to exercise caution when interpreting its outputs alongside professional medical consultation.
