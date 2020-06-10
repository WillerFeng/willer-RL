import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Atari_ConvNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, type:str):
        super(Atari_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32         , 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64         , 64, kernel_size=3, stride=1)
        
        self.fc1   = nn.Linear(7 * 7 * 64, 512)
        self.fc2   = nn.Linear(512, num_actions)
        
        if type == 'DQN':
            self.last_active = nn.ReLU()
        elif type == 'Policy':
            self.last_active = nn.softmax()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.last_active(self.fc2(x))
        return x
        
        
        
class Atari_Dueling_ConvNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, type:str):
        super(Atari_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32         , 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64         , 64, kernel_size=3, stride=1)
                   
        self.fc1   = nn.Linear(7 * 7 * 64, 512)
        self.fc2   = nn.Linear(512, num_actions)
        
        if type == 'DQN':
            self.last_active = nn.ReLU()
                   
        elif type == 'Policy':
            self.last_active = nn.softmax()
    
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
        
        
        
class FullyNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, type:str, hidden_size:int=256):        
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(inchannels , hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        
        
        if type == 'DQN':
            self.last_active = nn.ReLU()
                   
        elif type == 'Policy':
            self.last_active = nn.softmax()
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last_active(x)
        return x
        
        
        
class Dueling_FullyNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, type:str, hidden_size:int=256):       
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(inchannels , hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last_active(x)
        return x
                   