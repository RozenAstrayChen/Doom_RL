import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
'''
refer from https://github.com/nailo2c/a3c/blob/master/a3c.py
'''
class A3CNet(nn.Module):
    def __init__(self, num_inputs, action_sapce):
        super(A3CNet, self).__init__()
        # CNN
        # 84 x 84, 84+2 = 86 -> 42*42
        self.conv1 = nn.Conv2d(num_inputs, 32,kernel_size=3 ,stride=2, padding=1)
        # 42 x 42, 42+2 = 44 -> 22*22
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        # 22 x 22, 22+2 = 24 -> 12*12
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        # 12 x 12, 12+2 = 14 -> 6*6
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        # 6 x 6, 6+2 = 8 -> 3x3
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        #LSTM
        self.lstm = nn.LSTMCell(32*3*3,256)
        #actor-critic
        self.critic_linear = nn.Linear(256,1)
        self.actor_linear = nn.Linear(256, action_sapce)

        # weights init & normalized
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)

        # bias = 0
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        #train mode
        self.train()
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # flatten
        x = x.view(-1, 32*3*3)
        hx, cx = self.lstm(x, (hx, cx))
        x= hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

# refer from https://github.com/openai/universe-starter-agent
class SharedAdam(optim.Adam):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        
        super(SharedAdam, self).__init__(params, lr, betas, eps)
        
        # init to 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    # share adam's param
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    # update weight
    def step(self):
        loss = None
        
        for group in self.param_groups:
            for p in group['params']:
                # 檢查p是否有gradient，若沒有則進行下一個迴圈
                if p.grad is None: continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # 提取adam的參數
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # update first moment estimate & second moment estimate
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1
                
                # inplce mode of addcdiv
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return loss