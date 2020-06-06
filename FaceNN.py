import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import torchvision
import torchvision.transforms as transforms

def load_dataset():
    data_path = 'data/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    return train_loader

train_data = load_dataset()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 140)
        self.fc3 = nn.Linear(140, 84)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net()

if os.path.isfile('meinNetz.pt'):
    net = torch.load('meinNetz.pt')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(0):  # change if you want to train it more
    all_outputs = []
    all_losses = []
    running_loss = 0.0
    for i, data in enumerate(train_data, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_outputs.append(outputs[0].tolist())
        all_losses.append(loss .tolist())
    print(all_outputs)
    print(all_losses)
    print(epoch)

    torch.save(net, 'meinNetz.pt')

from PIL import Image

transform = torchvision.transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])



def image_loader(image_name):
    image = Image.open(image_name)
    image = transform(image).float()
    image = image.unsqueeze(0)
    return image


import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
rep = filedialog.askopenfilenames(parent=root,
                                      initialdir='data/test/',
                                      initialfile='',
                                      filetypes=[("JPEG", "*.jpg")])

if len(rep)<1: exit()

image = image_loader(rep[0])

output = net(image)
print(output)

classes = ['Gesicht', 'Kein Gesicht']
quality, index = torch.max(output, 1)
print(classes[index])

sigmoid = nn.Sigmoid()
quality = sigmoid(quality)
quality = quality[0].tolist()
print('Confidence: ', quality)