import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from time import time, sleep
import skimage.color
import skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt

from basic.config import *
from basic.process import *
from basic.env import *
from basic.shaping import *
from net.ac_net import ACNet
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor_Critic(Process):
    def __init__(self, map=map_health):

        print(torch.cuda.get_device_name(0))

        self.env = init_doom(map, visable=True)
        self.map = map

        n = self.env.get_available_buttons_size()
        self.action_dim = np.identity(n, dtype=int).tolist()
        self.action_num = n

        self.model = ACNet(self.action_num, resolution_dim * 4).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)

    def plot(self, frame_idx, rewards):
        #plt.figure(figsize=(20,5))
        #plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
        plt.plot(rewards)
        #plt.show()
        plt.legend()
        plt.pause(0.001)

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        '''
        print('\nn_v', next_value)
        print('rs', rewards)
        print('masks', masks, '\n')
        '''
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def test(self):
        stacked_frames = deque(
            [
                np.zeros((resolution[0], resolution[1]), dtype=np.int)
                for i in range(stack_size)
            ],
            maxlen=4)
        self.env.new_episode()
        s = self.env.get_state().screen_buffer
        s, stacked_frames = self.stack_frames(stacked_frames, s, True)
        done = False
        while not done:
            s = torch.FloatTensor(s).to(device)
            dist, _ = self.model.choose_action(s)
            a = dist.sample()
            take_action = self.action_dim[a.cpu().numpy()[0]]

            self.env.set_action(take_action)
            # frame skip 
            self.env.advance_action(5, True)

            done = self.env.is_episode_finished()
            if not done:
                n_s = self.env.get_state().screen_buffer
                n_s, stacked_frames = self.stack_frames(stacked_frames, n_s, False)
                s = n_s
        return self.env.get_total_reward()

    def train(self, horizon=200000):
        stacked_frames = deque(
            [
                np.zeros((resolution[0], resolution[1]), dtype=np.int)
                for i in range(stack_size)
            ],
            maxlen=stack_size)
        frame_idx = 0
        test_rewards = []

        self.env.new_episode()
        s = self.env.get_state().screen_buffer
        # got game variables
        var_old = self.env.get_state().game_variables
        s, stacked_frames = self.stack_frames(stacked_frames, s, True)
        done = False

        while frame_idx < horizon:
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(n_steps):

                if done:
                    self.env.new_episode()
                    s = self.env.get_state().screen_buffer
                    s, stacked_frames = self.stack_frames(
                        stacked_frames, s, True)
                    var_old = self.env.get_state().game_variables
                    print('new episode')
                
                s = torch.FloatTensor(s).to(device)

                dist, value = self.model.choose_action(s)
                a = dist.sample()
                take_action = self.action_dim[a.cpu().numpy()[0]]

                self.env.set_action(take_action)
                # frame skip 
                self.env.advance_action(5, True)
                r = self.env.get_last_reward()
                done = self.env.is_episode_finished()
                time.sleep(0.1)
                if not done:
                    # shapping
                    var_current = self.env.get_state().game_variables
                    r += shapping(var_old, var_current)

                var_old = var_current
                # store
                log_prob = dist.log_prob(a)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor([r]).unsqueeze(1).to(device))
                masks.append(
                    torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

                # judgement env is done
                if done:
                    n_s = np.zeros([resolution[0], resolution[1]])
                    n_s, stacked_frames = self.stack_frames(stacked_frames, n_s, False)
                else:
                    n_s = self.env.get_state().screen_buffer
                    n_s, stacked_frames = self.stack_frames(stacked_frames, n_s, False)

                s = n_s
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test() for _ in range(10)])
                    test_rewards.append(test_reward)
                    done = True
                    print('{} frame , test reward is : {}.'.format(
                        frame_idx, test_reward))
                    self.plot(frame_idx, test_rewards)

            n_s = torch.FloatTensor(n_s).to(device)
            _, n_v = self.model.choose_action(n_s)
            returns = self.compute_returns(n_v, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            
            advantage = returns - values
        
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.002 * entropy

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


agent = Actor_Critic()
agent.train()