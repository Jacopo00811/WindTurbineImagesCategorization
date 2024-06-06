# BSc Thesis Project: Wind Turbine Leading Edge Damage Categorization Using Deep Learning-Aided Image Recognition and Assessment of Its Effect on Aerodynamic Performance

Welcome to the GitHub repository for the BSc thesis project. This project is part of the final thesis submitted for the Bachelor of Science in Engineering at the Technical University of Denmark (DTU).

## Overview

This repository contains all the code used for the project, which consists of multiple neural networks capable of categorizing images of wind turbine blade damage into five different aerodynamic categories.

## Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
  - [Proposed CNNs](#proposed-cnns)
  - [Pre-trained Models](#pre-trained-models)
- [Training and Evaluation](#training-and-evaluation)
  - [Hyperparameter Search](#hyperparameter-search)
  - [Training](#training)
  - [Fine-Tuning](#fine-tuning)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Authors](#authors)
- [Acknowledgements and Info](#acknowledgements-and-info)

## Introduction

This project aims to create a medium-sized neural network capable of classifying images of leading edge damage automatically with high accuracy, thus enhancing the overall efficiency and reliability of wind turbine maintenance. The project also involves comparing the proposed CNNs with three pre-trained models using transfer learning and fine-tuning to fit the specific classification task.

## Dataset
The dataset consists of high-resolution images taken during turbine inspections, which were classified into one of five categories based on the severity of aerodynamic defects:
1. Category 1 - Lowest severity
2. Category 2
3. Category 3
4. Category 4
5. Category 5 - Highest severity

### Data Preprocessing
- **Data Extraction:** Images were extracted from PDF reports.
- **Data Cleaning:** Oversampling and class elimination techniques were used to balance the dataset.
- **Data Augmentation:** Various augmentation techniques were applied to enhance the dataset.
- **Data Split:** The dataset was split into training, validation, and test sets.
- **Principal Component Analysis (PCA):** PCA was applied to reduce dimensionality while preserving variance.

## Model Architecture
### Proposed CNNs
Two CNN architectures were proposed:
1. **Original-image CNN:** A basic CNN architecture using the original images.
2. **PCA CNN:** A CNN architecture using PCA-transformed images.

### Pre-trained Models
Three pre-trained models were fine-tuned for comparison:
1. Model MobileNetV3-large
2. Model ResNet152
3. Model VGG19 with batch normalization

## Training and Evaluation
### Hyperparameter Search
Hyperparameter search was performed using grid search and random search to optimize the model performance.

### Training
The training of the models took place on the DTU HPC servers using the provided GPUs.

### Fine-Tuning
The pre-trained models were fine-tuned to adapt them to the specific task of classifying wind turbine blade defects.

## Results
The performance of the models was evaluated based on:
- **Top 1 Classification Accuracy**
- **Top 2 Classification Accuracy**
- Analysis of Misclassified Images

The results showed that the proposed CNNs achieved high classification accuracy, with the PCA CNN performing slightly worse than the original-image CNN. The pre-trained models also demonstrated competitive performance and even higher accuracies.

## Future Improvements
Future improvements include:
- Exploring more advanced architectures like transformers
- Increasing the size and diversity of the dataset
- Implementing real-time classification capabilities

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- TensorBoard (optional) for training visualization

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/jacopo00811/WindTurbineImagesCategorization.git
cd WindTurbineImagesCategorization
```

## Authors

- Jacopo Ceccuti, BSc General Engineering Cyber System, DTU
- Andrea Fratini, BSc General Engineering Future Energy, DTU

## Acknowledgements and Info

Special thanks to the supervisors and for their guidance and support throughout this project. 
If you want to download the trained models don't hesitate to write me, I will be very happy to share them with you. For more details, please refer to the full [BSc Thesis](./BSc_Thesis.pdf).

