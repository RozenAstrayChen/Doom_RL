import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, available_actions_count):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4,stride=2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, available_actions_count)
    '''
        Input [60,108,3]
        conv1 -> [14,26,32]
        conv2 -> [6,12,64]
        flatten -> [4608]
    '''
    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4608)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),-1)
        return x
class ValueNetwork(nn.Module):
    def __init__(self, action_dim,init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4,stride=2)

        self.fcs1 = nn.Linear(4608, 512)
        self.fcs2 = nn.Linear(512, 128)
        # action
        self.fca1 = nn.Linear(action_dim, 128)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,1)

        self.fc4.weight.data.uniform_(-init_w, init_w)
        self.fc4.bias.data.uniform_(-init_w, init_w)
    '''
        Input [60,108,3]
        conv1 -> [14,26,32]
        conv2 -> [6,12,64]
        flatten -> [4608]
    '''
    def forward(self, state,action):
        action = action.view(action.size(0),1)
        #print('\n\n\n\t\tshape of action=',action.shape,'\n\n\n\t\t')
        s = F.relu(self.conv1(state))
        s = F.relu(self.conv2(s))
        s = s.view(s.size(0), -1)
        s = F.relu(self.fcs1(s))
        s = F.relu(self.fcs2(s))

        a = F.relu(self.fca1(action))
        x = torch.cat([s,a], dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


