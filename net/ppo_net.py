import math
import torch
import torch.nn as nn
import numpy as np
#from torch.distributions import Categorical
from torch.autograd import Variable
from basic.distrubtion import Categorical
from torch.nn import functional as F


class CNNPolicy(nn.Module):
    def __init__(self, state_dim, action_space):
        super(CNNPolicy, self).__init__()
        self.hidden_units = 72*64
        
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.hidden_units, 512)
        
        self.dist = Categorical(512, action_space)
        self.critic = nn.Linear(512, 1)
        

    def _flatten(self, x):
        flatten = x.view(-1, self.hidden_units)
        return flatten

    def forward(self, s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(-1, self.hidden_units)
        x = F.relu(self.fc1(x))
        v = self.critic(x)

        return v, x
    
    def action(self, s, deterministic=False):
        value, x = self.forward(s)
        action = self.dist.sample(x, deterministic)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)

        return value, action, action_log_probs

    def evaluate_actions(self, s, actions):
        actions = Variable(actions).long()
        value, x = self.forward(s)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)

        return value, action_log_probs, dist_entropy




