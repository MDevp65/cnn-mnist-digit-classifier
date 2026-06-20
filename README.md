# CNN Digit Classifier — MNIST

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.

## Overview

This notebook walks through the full deep learning pipeline:
- Loading and normalizing the MNIST dataset
- Building a CNN with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers
- Compiling and training the model
- Evaluating performance on the test set
- Visualizing training/validation accuracy and a confusion matrix
- A detailed breakdown of **why CNNs outperform plain ANNs on image data**

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## Model Architecture

```
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
→ Conv2D(64, 3x3, relu) → MaxPooling2D(2x2)
→ Flatten → Dense(128, relu) → Dropout(0.5)
→ Dense(10, softmax)
```

## Why CNN over ANN?

The notebook includes a full explanation of:
- Why feeding raw flattened pixels into an ANN causes overfitting
- How parameter explosion in dense layers leads to vanishing gradients
- How Conv2D, pooling, and weight sharing solve these problems

## Getting Started

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Open and run `DeepLearning_CNN_MNIST_annotated.ipynb` in Jupyter, VS Code, or Google Colab.

## Author

**Muhammad Muneeb**
AI/ML Engineer | SixAlps Agency
