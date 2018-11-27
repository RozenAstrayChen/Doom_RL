import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

    
class ACNet(nn.Module):

    def __init__(self, n_action, n_dim):
        super(ACNet, self).__init__()
        # CNN
        self.flatten_num = 64*3*3
        self.conv1 = nn.Conv2d(n_dim, 32, kernel_size=8, stride=4)
        # 84,84 -> 20,20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 20,20 -> 9,9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        # 9,9 -> 3,3

        #LSTM
        #self.lstm = nn.LSTMCell(32*3*3,256)
        self.fc1 = nn.Linear(self.flatten_num, 512)
        #actor-critic
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, n_action)

        
    def forward(self, inputs):
        #inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten
        x = x.view(-1, self.flatten_num)
        x = F.relu(self.fc1(x))
        # output
        value = self.critic_linear(x)
        probs = F.softmax(self.actor_linear(x), dim=1)
        #hx, cx = self.lstm(x, (hx, cx))

        return probs,  value

    def choose_action(self, x):
        x = x[np.newaxis, :]
        probs, value = self.forward(x)
        dist = Categorical(probs)

        return dist, value
    
    def get_v(self, x):
        _, value = self.forward(x)
        return value


