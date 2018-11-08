import numpy as np
import torch.nn as nn
from basic.distrubtion import Categorical
from torch.autograd import Variable
from torch.nn import functional as F
from basic.config import *


class Actor_Critic(nn.Module):
    def __init__(self, d_state, n_actions, lr=learning_rate):
        super(Actor_Critic, self).__init__()
        self.hidden_unit = 128*3*3


        self.conv1 = nn.Conv2d(d_state, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(self.hidden_unit, 512)

        self.v = nn.Linear(512, 1)
        self.dist = Categorical(512, n_actions)
    
    def forward(self, s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.hidden_unit)
        x = F.relu(self.fc1(x))

        v = self.v(x)

        return v, x

    def get_v(self, s):
        v, x = self.forward(s)
        return v
    
    def choose_action(self, s, deterministic=False):
        s = s[np.newaxis, :]
        v, x = self.forward(s)
        
        action = self.dist.sample(x, deterministic)
        return action

    def evaluate_action(self, s, actions):
        actions = Variable(actions).long()
        actions = actions[:, np.newaxis]
        value, x = self.forward(s)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)

        return value, action_log_probs, dist_entropy