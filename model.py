import torch
import torch.nn as nn
import torch.nn.functional as F

class EMNISTNet(nn.Module):
    def __init__(self, num_classes=26):
        super(EMNISTNet, self).__init__()
        
        # Enhanced convolutional layers with residual connections
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dropout layers with different rates
        self.dropout_conv = nn.Dropout2d(0.1)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        # Fully connected layers with better architecture
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)    
    def forward(self, x):
        # First convolutional block with residual-like connections
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.dropout_conv(x)
        
        # Second convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.dropout_conv(x)
        
        # Third convolutional block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 7x7 -> 3x3
        x = self.dropout_conv(x)
        
        # Adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)  # -> 4x4
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with skip connections
        x = self.dropout_fc1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc2(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
