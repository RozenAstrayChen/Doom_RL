import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ACNet(nn.Module):

    def __init__(self, available_actions_count):
        super(ACNet, self).__init__()
        # CNN
        
        self.conv1 = nn.Conv2d(3, 32,kernel_size=8 ,stride=4)
        # 84,84 -> 20,20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 20,20 -> 9,9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 9,9 -> 3,3

        #LSTM
        #self.lstm = nn.LSTMCell(32*3*3,256)
        self.fc1 = nn.Linear(64*3*3, 512)
        #actor-critic
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, available_actions_count)

        
 
    def forward(self, inputs):
        #inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # flatten
        x = x.view(-1, 64*3*3)
        x = F.relu(self.fc1(x))
        state_values = self.critic_linear(x)
        action_scores = self.actor_linear(x)
        #hx, cx = self.lstm(x, (hx, cx))

        return F.softmax(action_scores, dim=-1),   state_values

    def get_v(self, inputs):
        a_prob, v = self.forward(inputs)

        return v
