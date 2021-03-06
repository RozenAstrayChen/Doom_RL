from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from time import time, sleep
import skimage.color
import skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
from dqn import DQN
import matplotlib.pyplot as plt
'''
my tools
'''
from basic.env import *
from basic.config import *
from basic.memory import *
from net.q_net import Net
from basic.process import Process


'''
Double q need two weight
Fisrst weight is using chose action that dont need update
Second weight is using learn experience replay
.when update to minibatch which will clone to  First weight
'''


class DDQN(DQN):

    def __init__(self, map=map_basic):
        super().__init__(map)
        self.map = map
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]

        self.target_model = Net(len(actions)).to(self.device)
        self.eval_model = Net(len(actions)).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.target_model.parameters(),
                                             learning_rate)
        self.update_batch = 100
        self.counter = 0

    def choose_action(self, state, watch_flag=False):
        if self.eps > np.random.uniform() and watch_flag is False:
            action_index = np.random.randint(0, len(self.action_available) - 1)
            return action_index
        else:
            state = state.reshape([1, 3, resolution[0], resolution[1]])
            q = self.get_eval(state)
            m, index = torch.max(q, 1)
            action = index.data.cpu().numpy()[0]
            return action

    def get_eval(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state.to(self.device)
        return self.eval_model(state)

    def get_target(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state.to(self.device)
        return self.target_model(state)

    def learn_from_memory(self):
        self.counter += 1
        self.exploration_rate()
        if len(self.memory.buffer) > batch_size:

            batch = self.memory.get_sample(batch_size)
            s1 = np.array([each[0] for each in batch])
            a = np.array([each[1] for each in batch])
            r = np.array([each[2] for each in batch])
            s2 = np.array([each[3] for each in batch])
            isterminal = np.array([each[4] for each in batch])
            # using w which is eval
            target_q = self.get_target(s1).data.cpu().numpy()
            predict_q = self.get_eval(s2).data.cpu().numpy()

            predict_q = np.max(predict_q, axis=-1)
            target_q[np.arange(target_q.shape[0]),
                     a] = r + gamma * (1 - isterminal) * predict_q
            return self.propagation(s1, target_q)

    def propagation(self, state, target_q):
        s1 = torch.from_numpy(state)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        # change to gpu type
        s1 = s1.cuda(async=True)
        target_q = target_q.cuda(async=True)
        s1 = Parameter(s1.cuda(), requires_grad=True)
        if s1.grad is not None:
            s1.grad.data.zero_()
        '''
        forward propagation
        '''
        output = self.target_model(s1)
        '''
        get loos value
        '''
        loss = self.criterion(output, target_q).cuda()
        '''
        do back propagation
        '''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
        testing
        '''
        # self.plot_grad(s1.grad.data.cpu().numpy()[0])
        return loss

    def train_model(self, load=False, num=0, iterators=1):

        if load is True:
            self.target_model = self.load_model(double_dqn, num)
            self.eval_model = self.load_model(double_dqn, num)
        train_episodes_finished = 0
        rewards_collect = []
        for iterator in range(0, iterators):
            for epoch in range(total_episode):
                self.game.new_episode()
                train_scores = []
                s1 = self.preprocess(self.game.get_state().screen_buffer)
                s1 = self.frames_reshape(s1)
                for _ in range(learning_step_per_epoch):

                    action_index = self.choose_action(s1)
                    '''
                    self.game.set_action(
                        self.action_available[action_index])
                    # frame skip
                    self.game.advance_action(frame_skip)
                    '''
                    self.game.make_action(self.action_available[action_index], frame_skip)
                    reward = self.game.get_last_reward()

                    isterminal = self.game.is_episode_finished()
                    if not isterminal:
                        s2 = self.preprocess(
                            self.game.get_state().screen_buffer)
                        s2 = self.frames_reshape(s2)
                        self.memory.add_transition((s1, action_index, reward,
                                                    s2, isterminal))
                        s1 = s2
                    else:

                        s2 = np.zeros([3, resolution[0], resolution[1]]).astype(
                            np.float32)
                        self.memory.add_transition((s1, action_index, reward,
                                                    s2, isterminal))

                    loss = self.learn_from_memory()
                    # clone target net wights to eval
                    if self.counter % self.update_batch == 0:
                        self.eval_model.load_state_dict(
                            self.target_model.state_dict())

                    if self.game.is_episode_finished():
                        train_episodes_finished += 1
                        train_scores.append(self.game.get_total_reward())
                        break
                if (train_episodes_finished % 20 == 0):
                    print("%d training episodes played." %
                          train_episodes_finished)
                    rewards_collect.append(train_scores)
                    train_scores = np.array(train_scores)
                    print("Results: mean: %.1f +/- %.1f," %
                          (train_scores.mean(), train_scores.std()))
                    self.plot_durations(rewards_collect)
            self.save_model(double_dqn, iterator + 1, self.target_model)
        self.plot_save(rewards_collect)
        self.game.close()

    def watch_model(self, num):
        self.eval_model = self.load_model(double_dqn, num)
        self.target_model = self.load_model(double_dqn, num)
        self.game = init_doom(scenarios=self.map, visable=True)
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            state = self.preprocess(self.game.get_state().screen_buffer)
            state = self.frames_reshape(state)
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = self.frames_reshape(state)
                action_index = self.choose_action(state, watch_flag=True)
                self.game.set_action(self.action_available[action_index])
                self.game.advance_action(frame_skip)
                sleep(0.05)
                # reward = self.game.make_action(self.action_available[action_index])
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

        self.game.close()
