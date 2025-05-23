from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

bilal = transforms.ToTensor()

train_data = MNIST(root='.', train=True, download=True, transform=bilal)
test_data = MNIST(root='.', train=False, download=True, transform=bilal)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

"""
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Etiket : {label}")
plt.show()
"""

class BilalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1,784)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = BilalNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10)

for epoch in range(5):
    total_loss=0

    for images, labels in train_loader:
        outputs = model(images.view(-1,784))
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch : {epoch+1}, Loss : {total_loss:.4f}")

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.view(-1, 784))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct/total
print(f"Test doğruluk : {accuracy : .2f}")
