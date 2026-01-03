import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.gn1 = nn.GroupNorm(8, 64)  # 8 groups for 64 channels
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.gn2 = nn.GroupNorm(8, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.gn3 = nn.GroupNorm(8, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.gn1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.gn2(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.gn3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_flattened_size(self):
        x = torch.zeros(1, 1, 28, 28)
        self.eval()
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool1(x)
            x = self.gn1(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)
            x = self.gn2(x)
            x = F.relu(self.conv5(x))
            x = self.pool3(x)
            x = self.gn3(x)
            return x.view(1, -1).size(1)
