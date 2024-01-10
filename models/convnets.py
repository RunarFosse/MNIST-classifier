from torch import nn
from torch.nn import functional as F

class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 5), padding=1)   # Input starts as 28x28, becomes 26x26
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(5, 5), padding=1) # then 24x24
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 12x12
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=2) # 14x14
        self.conv4 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=2, padding=1) # 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 3x3

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (3 * 3), 144)
        self.fc2 = nn.Linear(144, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        # Convolutional part
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool2(out)

        # Fully connected part
        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers, 28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1) # 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 14x14
        self.conv2 = nn.Conv2d(8, 32, kernel_size=(3, 3)) # 12x12
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5)) # 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2) # 3x3

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (3 * 3), 86)
        self.fc2 = nn.Linear(86, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        # Convolutional part
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.pool2(out)

        # Fully connected part
        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class ConvNet3(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers, 28x28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1) # 28x28
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3)) # 12x12
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2) # 5x5

        # Fully connected layers
        self.fc1 = nn.Linear(32 * (5 * 5), 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Convolutional part
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        # Fully connected part
        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out