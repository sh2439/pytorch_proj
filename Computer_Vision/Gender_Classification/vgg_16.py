import torch
import torch.nn as nn
import torch.nn.functional as F

class vgg_16(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv1
        self.conv1_1 = nn.Conv2d(3, 64, (3,3), (1,1), padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,(3,3), (1,1), padding = 1)
        self.pool1 = nn.MaxPool2d((2,2))
        
        # Conv2
        self.conv2_1 = nn.Conv2d(64,128, (3,3),(1,1), padding = 1)
        self.conv2_2 = nn.Conv2d(128,128, (3,3),(1,1), padding = 1)
        self.pool2 = nn.MaxPool2d(2,2)
        
        # Conv3
        self.conv3_1 = nn.Conv2d(128,256,(3,3),(1,1), padding = 1)
        self.conv3_2 = nn.Conv2d(256,256, (3,3),(1,1), padding =1)
        self.conv3_3 = nn.Conv2d(256,256, (3,3),(1,1), padding = 1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        # Conv4
        self.conv4_1 = nn.Conv2d(256,512,(3,3),(1,1), padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,(3,3),(1,1), padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,(3,3),(1,1), padding = 1)
        self.pool4 = nn.MaxPool2d(2,2)
        
        # Conv5
        self.conv5_1 = nn.Conv2d(512,512,(3,3),(1,1), padding = 1)
        self.conv5_2 = nn.Conv2d(512,512,(3,3),(1,1), padding = 1)
        self.conv5_3 = nn.Conv2d(512,512,(3,3),(1,1), padding = 1)
        self.pool5 = nn.MaxPool2d(2,2)
        
        # FC layers
        self.fc6 = nn.Linear(4*4*512,100)
        self.fc7 = nn.Linear(100, 10)
        self.fc8 = nn.Linear(10,2)
        
    def forward(self,x):
        
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        # flatten x
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x
