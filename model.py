import torch.nn as nn
import torch.nn.functional as F
import torch

class BillNet(nn.Module):
    def __init__(self, num_classes):
        super(BillNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, 
                               kernel_size=3, stride=1, 
                               padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, 
                              kernel_size=3, stride=1, 
                              padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, 
                              kernel_size=3, stride=1, 
                              padding=0)
        
        self.pool = nn.MaxPool2d(2,2)
        f1_dim =  16 *  9 * 46     #####16*7*46                      
        self.fc1 = nn.Linear(f1_dim, 120)         
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x