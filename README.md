# 🧠 Handwritten Digit Recognition using TensorFlow

A beginner deep learning project that trains a fully connected neural network on the MNIST handwritten digit dataset and uses the trained model to classify custom handwritten digits.

This project was built to understand the fundamentals of neural networks, including data preprocessing, training, evaluation, model persistence, and inference.

---

## Project Overview

The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0–9).

After training, the model can predict unseen handwritten digits provided as 28×28 pixel images.

---

## Model Architecture

```text
Input (28 × 28)
        ↓
Flatten
        ↓
Dense (128, ReLU)
        ↓
Dense (128, ReLU)
        ↓
Dense (10, Softmax)
```

This architecture is a fully connected feedforward neural network.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib

---

## Dataset

MNIST is one of the most widely used introductory datasets in machine learning.

Dataset characteristics:

* 70,000 handwritten digit images
* 28×28 grayscale images
* Digits from 0–9
* Pre-labeled training and testing data

---

## Training Results

Model Performance:

```text
Test Accuracy: 97.81%
Test Loss: 0.1485
```

The model performs well on the MNIST test set and is capable of recognizing many custom handwritten digits.

Example custom test results:

```text
1 ✓
2 ✓
3 ✓
4 ✗
5 ✓
6 ✓
7 ✓
8 ✓
9 ✗

7 / 9 Correct
```

---

## Key Learnings

Through this project, I gained practical experience with:

* Neural Networks
* Activation Functions (ReLU, Softmax)
* Loss Functions
* Model Training
* Epochs and Optimization
* Accuracy Evaluation
* Saving and Loading Models
* Image Preprocessing

---

## Limitations

Although the model achieves high accuracy on MNIST, it can struggle with custom handwriting styles.

This is expected because the model uses fully connected dense layers, which do not capture spatial features in images as effectively as Convolutional Neural Networks (CNNs).

Future improvements include:

* Implementing a CNN architecture
* Improving image preprocessing
* Supporting multi-digit recognition
* Evaluating performance on larger custom datasets

---

## Conclusion

This project served as an introduction to deep learning and image classification using TensorFlow.

While the model is relatively simple, it provides a strong foundation for understanding how neural networks learn patterns from image data and make predictions on unseen inputs.
