# CIFAR-10 Image Classification — CNN from Scratch vs. Transfer Learning

Comparing a small Convolutional Neural Network trained entirely from scratch against a pretrained ResNet18 adapted via transfer learning, to quantify both the accuracy gain and the compute cost of each approach on a genuinely harder image dataset than MNIST.

> 🎓 This project follows directly from an earlier MLP vs. CNN comparison on MNIST, moving from grayscale digits to real, color objects — a meaningfully harder task that makes the benefit of transfer learning clearly visible.

## Overview

This project classifies CIFAR-10 images (10 classes of real-world objects — airplanes, cars, birds, cats, and more) using two different approaches: a small CNN built and trained entirely from scratch, and a ResNet18 pretrained on ImageNet with only its final classification layer retrained. Both were trained with data augmentation on the training set, and compared directly on test accuracy and total training time to capture the real practical tradeoff between the two approaches.

## Tools Used

- Python
- PyTorch, torchvision
- Pretrained ResNet18 (ImageNet weights)

## Dataset

**Source:** CIFAR-10 (via `torchvision.datasets`)

60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), pre-split into 50,000 training and 10,000 test images.

## Key Steps Performed

- Loaded CIFAR-10 with data augmentation (random horizontal flip, random rotation) applied to the training set only — never to test data, to keep evaluation on real, unmodified images
- Built a small CNN from scratch: two convolution + ReLU + max-pooling blocks, followed by fully-connected layers, trained on native 32×32 images
- Built a transfer learning model using a pretrained ResNet18: froze all pretrained layers, replaced and retrained only the final fully-connected layer for the 10 CIFAR-10 classes, with input images resized to 224×224 as ResNet's architecture expects
- Trained both models for the same number of epochs (5) with identical loss function (CrossEntropyLoss) and optimizer (Adam), to isolate the effect of architecture/pretraining rather than training setup
- Measured both test accuracy and total training time for a fair, practical comparison

## Results

| Model | Test Accuracy | Training Time |
|---|---|---|
| Small CNN (from scratch) | 54.12% | 62.2s |
| ResNet18 (transfer learning) | **79.70%** | 506.4s |

## Key Insights

- **Transfer learning produced a massive accuracy gain** — nearly 26 percentage points higher than the from-scratch CNN — despite only training the final classification layer. ResNet18's early layers, pretrained on millions of diverse ImageNet images, already encode general-purpose visual features (edges, textures, shapes) that transfer directly to a new, unrelated classification task with minimal additional training.
- **That accuracy came at a real compute cost**: ResNet18 took roughly 8x longer to train, driven mostly by the 224×224 input size ResNet's architecture requires versus CIFAR-10's native 32×32 resolution, plus the deeper network itself even with most layers frozen. This is a genuine, practical tradeoff — not a free win — and a consideration that matters when choosing an architecture under real time or compute constraints.
- **The small CNN's 54% is a reasonable result in context, not a failure**: CIFAR-10 is meaningfully harder than MNIST (color images, real-world object variation, low resolution), and a simple two-layer convolutional network trained for only 5 epochs from a random initialization has no prior knowledge to draw on — it has to learn everything, including basic edge and texture detection, from scratch within those 5 epochs alone.
- **This result illustrates why transfer learning is the default approach for most real-world image classification tasks**: training a capable CNN from scratch typically requires far more data, far more epochs, and far more compute than most practical projects have available — reusing a pretrained network's general visual knowledge is almost always the more efficient path, provided the added compute cost for larger input images is acceptable.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install torch torchvision
   ```
3. Open the script/notebook and run all cells in order (CIFAR-10 downloads automatically via `torchvision`, and ResNet18's pretrained weights download automatically on first use)
4. Note: training ResNet18 at 224×224 resolution is significantly slower than the small CNN — a GPU is recommended if available, though the script runs on CPU as well

## What I'd Improve Next

- Unfreeze and fine-tune the later layers of ResNet (not just the final layer) to see if accuracy improves further, at the cost of additional training time
- Try a smaller pretrained model (e.g. MobileNet) to see if it offers a better accuracy-to-compute-time tradeoff than ResNet18 for this dataset's small image size
- Train the from-scratch CNN for more epochs to see how much of the accuracy gap closes with additional training time, rather than architecture alone

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning, and builds directly on an earlier MLP vs. CNN comparison on the MNIST dataset.