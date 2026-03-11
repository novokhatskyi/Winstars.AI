# WinStars AI

This repository contains two independent test tasks related to Machine Learning and AI.

## Task 1 — MNIST Image Classification + OOP
The first task focuses on handwritten digit classification using the **MNIST** dataset.

It includes three models:
- Random Forest
- Feed-Forward Neural Network
- Convolutional Neural Network

The solution is implemented using **Object-Oriented Programming (OOP)** principles:
- a common classifier interface,
- separate classes for each model,
- a wrapper class that selects the algorithm (`rf`, `nn`, `cnn`).

A demo notebook is included to show model training, prediction, and result comparison.

## Task 2 — NER + Computer Vision Pipeline
The second task combines **Natural Language Processing** and **Computer Vision** in one pipeline.

The solution includes:
- a **NER model** for extracting the animal name from text,
- an **image classification model** for recognizing the animal in an image,
- a pipeline that compares both outputs and returns a boolean result.

The demo notebook shows how the pipeline processes text and image inputs and decides whether they match.

## Repository Structure

```bash
WinStars_AI/
├── mnist_task_1/
└── NER_CV/
```

Each task contains its own detailed `README.md` with setup instructions, project structure, and usage examples.
