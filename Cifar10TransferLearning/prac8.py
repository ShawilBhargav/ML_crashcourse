import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Augmentation: only on train data
train_transfrom_small = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
test_transform_small = transforms.ToTensor()

train_data_small = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transfrom_small)
test_data_small = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform_small)

train_loader_small = DataLoader(train_data_small, batch_size=64, shuffle=True)
test_loader_small = DataLoader(test_data_small, batch_size=64, shuffle=True)

# separate loaders for resnet18, cuz requires larger imgs
train_transform_resized = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
test_transform_resized = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

train_data_resized = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform_resized)
test_data_resized = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform_resized)

train_loader_resized = DataLoader(train_data_resized, batch_size=64, shuffle=True)
test_loader_resized = DataLoader(test_data_resized, batch_size=64, shuffle=True)

# 2. Model-1: Training a CCN from scratch
class smallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Model-2: Pretrained CNN model
def build_resnet():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# 4. Shared Training / Evaluations
def train_model(model, train_loader, epochs, lr=0.01, only_train_fc=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params_to_train = model.fc.parameters() if only_train_fc else model.parameters()
    optimizer = torch.optim.Adam(params_to_train, lr=lr)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f} seconds")
    return elapsed

def eval_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

# Running both models
print("\n ---Training Small CNN from scratch---")
small_cnn = smallCNN()
small_cnn_time = train_model(small_cnn, train_loader_small, epochs=5)
small_cnn_acc = eval_model(small_cnn, test_loader_small)

print("\n ---Training ResNet18 (transfer learning)---")
resnet = build_resnet()
resnet_time = train_model(resnet, train_loader_resized, epochs=5)
resnet_acc = eval_model(resnet, test_loader_resized)

print("\n ---Comparison between two---")
print(f"Small CNN -> Accuracy: {small_cnn_acc:.2f}%, Time: {small_cnn_time:.1f}s")
print(f"ResNet18 -> Accuracy: {resnet_acc:.2f}%, Time: {resnet_time:.1f}s")