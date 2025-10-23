import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose a dataset -- MNIST for example
dataset = datasets.MNIST(root='data', train=True, download=True)

# Set how the input images will be transformed
dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307,], std=[0.3081,])
])

# Create a data loader
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=1)

# Show the shape of a batch.

print("The shape of one batch is {}".format((next(iter(train_loader)))[0].size()))

"""
Define the Model. The requirements are:
    Define the first convolutional layer with channel size = 5, kernel size = 3 and stride = 2, padding = 1.
    Define the second convolutional layer with channel size = 8, kernel size = 3 and stride = 1, padding = 1.
    Use max pooling layer with stride = 2 between the two convolution layers.
    Define the FC layer(s) and finally return a tensor with shape torch.Size([256, 10]). (Use torch.view to reshape tensors. You can try any number of FC layers).
    Use ReLU activation between any two layers.
"""

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # Call parent class's constructor

        #############################################################################
        # TODO: Define the model structure                                          #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(8 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Define forward function                                             #
        #############################################################################
        y = self.conv1(x)
        y = self.relu(y)

        y = self.pool(y)

        y = self.conv2(y)
        y = self.relu(y)

        # Flatten the tensor for FC layers
        y = y.view(y.size(0), -1)  # shape: [batch_size, 8*7*7]

        y = self.relu(self.fc1(y))
        y = self.fc2(y)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return y



model = SimpleCNN().to(device)  # You may change 'device' here !!!
#
#
#
#############################################################################
# TODO: Define the criterion to be Cross Entropy Loss.                      #
#       Define the optimizer to be SGD with momentum factor 0.9             #
#       and weight_decay 5e-4.
# You may change the learning rate.                                      #
#############################################################################
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


#############################################################################
#                          END OF YOUR CODE                                 #
#############################################################################

def train(epoch):
    model.train()  # Set the model to be in training mode
    for batch_index, (inputs, targets) in enumerate(train_loader):
        # Forward
        # You may change 'device' of inputs here !!!
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if batch_index % 10 == 0:
            print('epoch {}  batch {}/{}  loss {:.3f}'.format(
                epoch, batch_index, len(train_loader), loss.item()))

        # Backward
        optimizer.zero_grad()  # Set parameter gradients to zero
        loss.backward()        # Compute (or accumulate, actually) parameter gradients
        optimizer.step()       # Update the parameters

# Choose a dataset -- MNIST for example
dataset = datasets.MNIST(root='data', train=False, download=True)

# Set how the input images will be transformed
dataset.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, ], std=[0.3081, ])
])

# Create a data loader
test_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

def test(dataloader):
    model.eval()

    # Evaluate your model on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            # images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy and loss
    accuracy = correct / total
    print('Accuracy:', accuracy)

for epoch in range(0, 6):
    train(epoch)
    # You may validate model here
test(test_dataloader)
