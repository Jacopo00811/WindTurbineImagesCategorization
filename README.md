# BSc Thesis Project: Wind turbine leading edge damage: categorization using deep learning aided image recognition and assessment of its effect on aerodynamic performance

Welcome to the GitHub repository for the BSc thesis project. This project is part of the final thesis submitted for the Bachelor of Science in Engineering at the Technical University of Denmark (DTU).

## Overview

This repository contains all the code used for the project, which consist in multiple neural networks capable of categorizing images of wind turbine blades into 5 different aerodynamics damage categories.

## Contents

- [Introduction](#introduction)
- [Data](#data)
  - [Data Extraction](#data-extraction)
  - [Data Cleaning](#data-cleaning)
  - [Data Augmentation](#data-augmentation)
  - [Data Split](#data-split)
  - [Principal Component Analysis](#principal-component-analysis)
- [Models](#models)
  - [Proposed CNNs](#proposed-cnns)
  - [Pre-trained Models](#pre-trained-models)
- [Hyperparameter Tuning and Training](#hyperparameter-tuning)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Training](#training)
  - [Fine Tuning](#fine-tuning)
- [Results](#results)
  - [Top 1 Classification Accuracy](#top-1-classification-accuracy)
  - [Top 2 Classification Accuracy](#top-2-classification-accuracy)
  - [Misclassified Samples](#misclassified-samples)
- [Conclusions and Future Work](#conclusions-and-future-work)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## Introduction

This project aims to create a medium-sized neural network capable of classifying these images automatically with high accuracy, thus enhancing the overall efficiency and reliability of wind turbine maintenance. The project also involves comparing the proposed CNNs with three pre-trained models using transfer learning and fine-tuning to fit the specific classification task.

## Dataset
The dataset consists of high-resolution images taken during turbine inspections, which were classified into one of five categories based on the severity of structural defects:
1. Category I - Lowest severity
2. Category II
3. Category III
4. Category IV
5. Category V - Highest severity

### Data Preprocessing
- **Data Extraction:** Images were extracted from PDF reports.
- **Data Cleaning:** Oversampling and class elimination techniques were used to balance the dataset.
- **Data Augmentation:** Various augmentation techniques were applied to enhance the dataset.
- **Data Split:** The dataset was split into training, validation, and test sets.
- **Principal Component Analysis (PCA):** PCA was applied to reduce dimensionality while preserving variance

## Model Architecture
### Proposed CNNs
Two CNN architectures were proposed:
1. **Original-image CNN:** A basic CNN architecture using the original images.
2. **PCA CNN:** A CNN architecture using PCA-transformed images.

### Pre-trained Models
Three pre-trained models were fine-tuned for comparison:
1. Model A
2. Model B
3. Model C

## Training and Evaluation
### Hyperparameter Tuning
Hyperparameters were tuned using grid search to optimize the model performance.

### Training
Models were trained on the training set and validated on the validation set. Training involved adjusting weights using backpropagation and minimizing the loss function.

### Fine-Tuning
The pre-trained models were fine-tuned to adapt them to the specific task of classifying wind turbine blade defects.

## Results
The performance of the models was evaluated based on:
- **Top 1 Classification Accuracy**
- **Top 2 Classification Accuracy**
- Analysis of Misclassified Images

The results showed that the proposed CNNs achieved high classification accuracy, with the PCA CNN performing slightly worse than the original-image CNN. The pre-trained models also demonstrated competitive performance.

## Future Improvements
Future improvements include:
- Exploring more advanced architectures
- Increasing the size and diversity of the dataset
- Implementing real-time classification capabilities

## Getting Started
### Prerequisites
- Python 3.x
- TensorFlow or PyTorch
- Other dependencies listed in `requirements.txt`

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/wind-turbine-blade-classification.git
cd wind-turbine-blade-classification
pip install -r requirements.txt

## Conclusions and Future Work

The project concludes with a summary of the findings and suggestions for future improvements to the model. Potential directions for future work include exploring more advanced data augmentation techniques and testing other machine learning algorithms.

## Authors

- Jacopo Ceccuti, MSc Civil Engineering, DTU
- Andrea Fratini, MSc Civil Engineering, DTU

## Acknowledgements

Special thanks to our supervisors and colleagues at DTU for their guidance and support throughout this project.

---

For more details, please refer to the full [BSc Thesis](./BSc_Thesis.pdf).

Feel free to explore the code and provide feedback or contribute to the project!
