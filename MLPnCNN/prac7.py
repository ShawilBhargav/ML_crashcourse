import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# MLP (Multi-Layer Perceptrons) or Feedforward Neural Network
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

digits = DigitClassifier()
# print(digits)

# this combines softmax n cross entropy calc.
criterion = nn.CrossEntropyLoss()
# smarter and more adaptive than gradient descent
optimizer = torch.optim.Adam(digits.parameters(), lr=0.001)

# TRAINING
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = digits(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# PRINTING ACCURACY
correct = 0
total = 0
digits.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = digits(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"test Accuracy: {100*correct/total:.2f}%")

# Visualising
# digits.eval()
# wrong_imgs = []
# wrong_preds = []
# wrong_labels = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = digits(images)
#         _, predicted = torch.max(outputs, 1)
#         mismatched = predicted != labels
#         wrong_imgs.extend(images[mismatched])
#         wrong_preds.extend(predicted[mismatched])
#         wrong_labels.extend(labels[mismatched])
#         if len(wrong_imgs) >= 9:
#             break
#
# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# for i, ax in enumerate(axes.flat):
#     if i < len(wrong_imgs):
#         ax.imshow(wrong_imgs[i].squeeze(), cmap='gray')
#         ax.set_title(
#             f"True: {wrong_labels[i].item()}, "
#             f"Pred: {wrong_preds[i].item()}"
#         )
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# CNN (Convolution Neural Network)
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNNClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

# TRAINING
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    # print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# PRINTING ACCURACY
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {(100 * correct / total):.4f}%")

# Visualising
cnn_model.eval()
wrong_imgs = []
wrong_preds = []
wrong_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        mismatched = predicted != labels
        wrong_imgs.extend(images[mismatched])
        wrong_preds.extend(predicted[mismatched])
        wrong_labels.extend(labels[mismatched])
        if len(wrong_imgs) >= 9:
            break

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    if i < len(wrong_imgs):
        ax.imshow(wrong_imgs[i].squeeze(), cmap='gray')
        ax.set_title(
            f"True: {wrong_labels[i].item()}, "
            f"Pred: {wrong_preds[i].item()}"
        )
    ax.axis('off')
plt.tight_layout()
plt.show()
