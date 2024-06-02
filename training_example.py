import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Determine the training device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training device is {device}.")

# Data Preparation

transform = transforms.Compose([
    transforms.ToTensor(),	# Converts a PIL Image or numpy.ndarray to a tensor.
    transforms.Normalize((0.5,), (0.5,))	#  Normalizes a tensor with mean and standard deviation.
])

torch.manual_seed(1024) # Fix random seed

# Downloads the Fashion-MNIST dataset.
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# Provides an iterable over the given dataset.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Let us print some samples in the dataset
import matplotlib.pyplot as plt
import numpy as np

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))

# Print labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print('Categories for the first row:')
print(', '.join(f'{classes[labels[j]]}' for j in range(8)))

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleCNN()
net = net.to(device)  # move the model to the training device
net

# Define the loss function
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
import time

for epoch in range(10):  # loop over the dataset 10 times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        start_time = time.time()
        inputs, labels = data	# get the training data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()	# clear the gradients

        outputs = net(inputs)	# forward computation
        loss = criterion(outputs, labels)	# compute the loss
        loss.backward()	# backward computation to derive the gradients
        optimizer.step()	# apply the gradients

        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(
                f'[Epoch {epoch + 1}, Batch {i + 1}] '
                f'loss: {running_loss / 100:.3f}, '
                f'step time: {(time.time() - start_time) * 1000:.2f}ms'
            )
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Let's visualize the results

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Print images
imshow(torchvision.utils.make_grid(images[:8].cpu()))
print('GroundTruth:\t', ',\t'.join(f'{classes[labels[j]]}' for j in range(8)))

# Predict
outputs = net(images[:8])
_, predicted = torch.max(outputs, 1)

print('Predicted:\t', ',\t'.join(f'{classes[predicted[j]]}' for j in range(8)))