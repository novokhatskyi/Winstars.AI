# MNIST Image Classification with OOP

## Project Overview

This project implements handwritten digit classification on the **MNIST** dataset using **Object-Oriented Programming (OOP)** principles.

Three classification models are implemented:

- **Random Forest**
- **Feed-Forward Neural Network**
- **Convolutional Neural Network**

Each model is implemented as a separate class that follows a common interface: `MnistClassifierInterface`.

A wrapper class, `MnistClassifier`, provides a unified entry point and allows selecting the algorithm through a single parameter:

- `rf` - Random Forest
- `nn` - Feed-Forward Neural Network
- `cnn` - Convolutional Neural Network

---

## Project Structure

```text
mnist_class/
│
├── demo.ipynb
├── README.md
├── requirements.txt
├── data/
└── source/
    ├── __init__.py
    ├── interface.py
    ├── rf_classifier.py
    ├── nn_classifier.py
    ├── cnn_classifier.py
    └── mnist_classifier.py
```

---

## Task Description

The goal of this project is to classify handwritten digits from the MNIST dataset using three different approaches:

1. **Random Forest**
2. **Feed-Forward Neural Network**
3. **Convolutional Neural Network**

The implementation follows an OOP design:

- `MnistClassifierInterface` defines two abstract methods:
  - `train()`
  - `predict()`
- each classifier is implemented in a separate class:
  - `RandomForestMnistClassifier`
  - `FeedForwardNN`
  - `CNNClassifier`
- `MnistClassifier` acts as a unified wrapper around all three models.

---

## Implemented Models

### 1. Random Forest
A classical machine learning model based on an ensemble of decision trees.

### 2. Feed-Forward Neural Network
A fully connected neural network for digit classification.

### 3. Convolutional Neural Network
A CNN model designed specifically for image classification.

---

## Technologies Used

- Python
- NumPy
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## Dataset

The project uses the **MNIST** dataset from `torchvision.datasets.MNIST`.

Dataset properties:

- image size: `28 x 28`
- image type: grayscale
- number of classes: `10`
- labels: digits from `0` to `9`

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If needed, you can also install the main libraries manually:

```bash
pip install numpy torch torchvision scikit-learn matplotlib jupyter
```

---

## How to Run

### Demo notebook
The main demonstration of the project is provided in:

```text
demo.ipynb
```

Run Jupyter Notebook from the project root:

```bash
jupyter notebook
```

Then open `demo.ipynb` and execute the cells step by step.

---

## Example Usage

### Import the wrapper class

```python
from source.mnist_classifier import MnistClassifier
```

### Create a model

```python
model = MnistClassifier(algorithm="cnn")
```

Available values for `algorithm`:

- `"rf"`
- `"nn"`
- `"cnn"`

### Train the model

For CNN or NN:

```python
model.train(X_train, y_train, num_epochs=5, batch_size=64, learning_rate=0.001)
```

For Random Forest:

```python
model.train(X_train, y_train)
```

### Predict

```python
y_pred = model.predict(X_test)
```

---

## Demo Workflow

The demo notebook includes:

- loading the MNIST dataset
- visualizing sample images
- selecting a classification model
- training the selected model
- predicting labels for the test set
- evaluating model accuracy
- measuring execution time

---

## Input Handling

### Random Forest and Feed-Forward Neural Network
These models support the following input formats:

- `[N, 784]`
- `[N, 28, 28]`
- `[N, 1, 28, 28]`

The input is flattened internally when needed.

### CNN
The CNN classifier supports the following input formats:

- `[N, 784]`
- `[N, 28, 28]`
- `[N, 1, 28, 28]`

The input is reshaped internally to match CNN requirements.

---

## Edge Case Handling

The project includes basic handling for edge cases such as:

- unsupported algorithm name
- unexpected input shape
- automatic reshaping or flattening of valid MNIST input formats

Example:

```python
MnistClassifier(algorithm="invalid")
```

Output:

```python
ValueError: Unsupported model type: invalid
```

---

## Results

This project allows comparing three different classification approaches on the same dataset within a unified OOP architecture.

The demo output includes:

- model accuracy
- execution time
- prediction results

---

## Notes

- The `source/` directory contains project modules and is not intended to be run as standalone scripts.
- The recommended way to demonstrate the project is through `demo.ipynb`.
- Relative imports are used inside the `source` package.

---

## Author

Oleksandr Novokhatskyi
