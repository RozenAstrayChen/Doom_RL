import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    # 除以norm 2，再乘以std來控制拉伸的長度
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out
    
# Xavier initialization
def weights_init(m):
    classname = m.__class__.__name__
    # 對捲積層做initialization
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    
    # 對FC做initialization
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class ACNet(nn.Module):

    def __init__(self, available_actions_count):
        super(ACNet, self).__init__()
        # CNN
        
        self.conv1 = nn.Conv2d(3, 32,kernel_size=8 ,stride=4)
        # 84,84 -> 20,20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 20,20 -> 9,9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        # 9,9 -> 3,3

        #LSTM
        #self.lstm = nn.LSTMCell(32*3*3,256)
        self.fc1 = nn.Linear(64*3*3, 512)
        #actor-critic
        self.critic_linear = nn.Linear(512,1)
        self.actor_linear = nn.Linear(512, available_actions_count)

        # weights init & normalized
        #self.apply(weights_init)
        #self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        #self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)

        # bias = 0
        #self.actor_linear.bias.data.fill_(0)
        #self.critic_linear.bias.data.fill_(0)
        #self.lstm.bias_ih.data.fill_(0)
        #self.lstm.bias_hh.data.fill_(0)
 
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
