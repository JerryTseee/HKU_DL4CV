from __future__ import annotations

import torch
import torchvision
from torch import optim, nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# from __future__ import annotations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from math import fabs
transform = v2.Compose(
    [
        v2.ToTensor(),
        #############################################################################
        # TODO:                                                                     #
        # 1. update mean and std given the statistics of training data              #
        # 2. set your own data augmentation combinations                            #
        #############################################################################
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
    ]
)

train_dataset = datasets.CIFAR10("data", transform=transform, train=True, download=True)
test_dataset = datasets.CIFAR10("data", transform=transform, train=False, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

#############################################################################
# TODO:                                                                     #
# 1. Use a pre-defined model in torchvision.models                          #
# 2. Define the criterion                                                   #
# 3. Define the optimizer                                                   #
# 4. Adjust the hyperparameters                                             #
#############################################################################

model = torchvision.models.resnet18(pretrained=False, num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) # This is a learning rate scheduler that gradually decreases the learning rate

#############################################################################
#                          END OF YOUR CODE                                 #
#############################################################################

def train_step(input: Tensor, label: Tensor):
    #############################################################################
    # TODO: implement a training step                                           #
    #############################################################################
    model.train() # set into train mode
    input, label = input.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    return loss, output


@torch.no_grad()
def test_step(input: Tensor, label: Tensor):
    #############################################################################
    # TODO: implement a testing step                                           #
    #############################################################################
    model.eval()
    input, label = input.to(device), label.to(device)
    output = model(input)
    loss = criterion(output, label)

    return loss, output


def train_epoch(dataloader):
    total = 0
    correct = 0

    for i, (input, label) in enumerate(dataloader):
        loss, output = train_step(input, label)
        #############################################################################
        # TODO: implement accuracy calculation                                      #
        # You may add additional metrics to better evaluate your model              #
        #############################################################################
        label = label.to(device)
        _, predicted = output.max(1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        if i % 100 == 0:
            accuracy = 100 * correct / total
            print(f"Step {i}: Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################


def test_epoch(dataloader):
    correct = 0
    total = 0
    for i, (input, label) in enumerate(dataloader):
        loss, output = test_step(input, label)
        #############################################################################
        # TODO: implement accuracy calculation                                      #
        # You may add additional metrics to better evaluate your model              #
        #############################################################################
        label = label.to(device)
        _, predicted = output.max(1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        if i % 100 == 0:
            accuracy = 100 * correct / total
            print(f"Step {i}: Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

for epoch in range(20):
    print("training phase:")
    train_epoch(train_dataloader)
    print("testing phase:")
    test_epoch(test_dataloader)
