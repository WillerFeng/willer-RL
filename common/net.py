import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Atari_ConvNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, net_type:str):
        super(Atari_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32         , 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64         , 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_space)
        
        if   net_type == 'value':
            self.last_active = nn.ReLU()
        elif net_type == 'policy':
            self.last_active = nn.softmax(dim=1)
        else:
            raise ValueError("Undefined net type")
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        action = F.relu(self.fc1(x))
        action = self.last_active(self.fc2(action))
        return action, value
        
        
        
class Atari_Dueling_ConvNet(nn.Module):
    
    def __init__(self, in_channels:int, action_space:int, net_type:str):
        super(Atari_Dueling_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32         , 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64         , 64, kernel_size=3, stride=1)
                   
        self.fc_action1 = nn.Linear(7 * 7 * 64, 512)
        self.fc_action2 = nn.Linear(512, action_space)
        
        self.fc_value1  = nn.Linear(7 * 7 * 64, 512)
        self.fc_value2  = nn.Linear(512, 1)
        
        if   net_type == 'value':
            self.last_active = nn.ReLU()
        elif net_type == 'policy':
            self.last_active = nn.softmax(dim=1)
        else:
            raise ValueError("Undefined net type")
    
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        action = F.relu(self.fc_action1(x))
        action = self.last_active(self.fc_action2(action))
        
        value = F.relu(self.fc_value1(x))
        value = self.fc_value2(value)
        return action, value
        
        
        
class FullyNet(nn.Module):
    
    def __init__(self, state_space:int, action_space:int, net_type:str, hidden_size:int=256):        
        super(FullyNet, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)
        
        
        if   net_type == 'value':
            self.last_active = nn.ReLU()
        elif net_type == 'policy':
            self.last_active = nn.softmax(dim=1)
        else:
            raise ValueError("Undefined net type")
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.last_active(x)
        return action
        
        
        
class Dueling_FullyNet(nn.Module):
    
    def __init__(self, state_space:int, action_space:int, net_type:str, hidden_size:int=256):       
        super(Dueling_FullyNet, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.fc_value  = nn.Linear(hidden_size, 1)
        self.fc_action = nn.Linear(hidden_size, action_space)
        
        if   net_type == 'value':
            self.last_active = nn.ReLU()
        elif net_type == 'policy':
            self.last_active = nn.Softmax(dim=1)
        else:
            raise ValueError("Undefined net type")
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action = self.last_active(self.fc_action(x))
        value  = self.fc_value(x)
        return action, value
                   