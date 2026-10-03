# Handwritten Digit Classification — MLP vs. CNN (MNIST)

Building a neural network from first principles — architecture, forward propagation, loss, backpropagation, and gradient descent all implemented and understood explicitly — then comparing a basic fully-connected network against a Convolutional Neural Network to see how an architecture built for spatial data outperforms a generic one.

> 🎓 This is my first deep learning project, and the first time I've built a model from the ground up in PyTorch rather than calling a single `.fit()` method. The goal here was to understand every stage of how a neural network learns, not just reach a high accuracy number.

## Overview

This project classifies handwritten digits (0-9) from the MNIST dataset using two different neural network architectures: a basic Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN). Both were trained from scratch in PyTorch, with the full training loop (forward propagation, loss calculation, backpropagation, gradient descent) written and understood explicitly rather than abstracted away. Misclassified digits from both models were visually reviewed to check whether remaining errors reflected genuine model limitations or inherently ambiguous handwriting.

## Tools Used

- Python
- PyTorch, torchvision
- Matplotlib

## Dataset

**Source:** MNIST (via `torchvision.datasets`)

70,000 grayscale images (28×28 pixels) of handwritten digits, pre-split into 60,000 training and 10,000 test images.

## Key Steps Performed

- Loaded MNIST via `torchvision` and batched it using `DataLoader` (batch size 64)
- Built a baseline MLP: `Flatten → Linear(784, 128) → ReLU → Linear(128, 10)`
- Built a CNN: two convolution + ReLU + max-pooling blocks, followed by fully-connected layers: `Conv(1→16) → ReLU → Pool → Conv(16→32) → ReLU → Pool → Flatten → Linear(1568, 128) → ReLU → Linear(128, 10)`
- Trained both models for 5 epochs using `CrossEntropyLoss` and the Adam optimizer, writing the full training loop explicitly (`zero_grad` → forward pass → loss → `backward()` → `optimizer.step()`) rather than using a high-level training API
- Evaluated both models on the held-out test set
- Visually inspected a sample of misclassified digits from each model to judge whether errors reflected genuine model weakness or inherently ambiguous handwriting

## Results

| Model | Test Accuracy | Error Rate |
|---|---|---|
| MLP (fully-connected) | 97.60% | 2.40% |
| CNN | 98.70% | 1.30% |

## Key Insights

- **The CNN reduced the error rate by roughly half** compared to the MLP (2.40% → 1.30%), despite both models being trained identically otherwise (same loss function, optimizer, epochs, and data). The only difference was architecture — convolution and pooling layers that respect the image's spatial structure, versus a generic fully-connected network that flattens the image and treats every pixel as independent from the start.
- **Both models' remaining errors, on visual inspection, were genuinely ambiguous handwriting** — digits that would plausibly be misread by a human too — rather than clearly legible digits the models should have gotten right. This suggests both architectures are operating near their practical ceiling on this dataset, and further gains would likely require either more training data, data augmentation, or a fundamentally different approach (e.g. ensembling multiple CNNs) rather than simply more epochs.
- **Understanding the full training loop explicitly** (rather than using a black-box `.fit()`) made it possible to reason clearly about why each architectural choice mattered — for example, the final layer of both networks deliberately has no activation function, since `CrossEntropyLoss` applies softmax internally, a detail that's easy to miss when only using high-level APIs.
- **This project marks a direct conceptual bridge from classical ML**: the weighted-sum-plus-bias structure of a single neuron is mathematically identical to Linear Regression, and the training loop (forward pass → loss → gradient computation → weight update) is the same gradient descent process used throughout earlier projects, just computed across many more parameters via backpropagation instead of a single direct gradient calculation.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install torch torchvision matplotlib
   ```
3. Open the notebook in Jupyter Notebook or Google Colab
4. Run all cells in order (MNIST downloads automatically via `torchvision`)

## What I'd Improve Next

- Train for more epochs with learning rate scheduling to see how far accuracy can realistically be pushed
- Add data augmentation (small rotations, shifts) to see if it helps the CNN generalize to the remaining ambiguous cases
- Visualize the CNN's learned convolutional filters directly, to see what patterns (edges, curves) the early layers are actually detecting

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning. It marks the first project in the deep learning phase of that path.