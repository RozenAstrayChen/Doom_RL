import tensorflow as tf
import numpy as np
import sys
import itertools as it
from basic.env import *
from basic.config import *
from basic.process import *
from basic.rollout import *
from ppo_batch import *
import time


class Agent(Process):
    def __init__(self, map=map_health):
        self.map = map
        self.env = init_doom(map, visable=False)
        self.sess = tf.Session()

        n = self.env.get_available_buttons_size()
        self.action_dim = np.identity(n, dtype=int).tolist()
        self.action_num = n

        self.model = PPO_batch(self.sess, self.action_num)
        self.rollout = Rollout(batch=mini_batch)

    def enjoy(self, load=False):
        #self.env = restart_doom(self.env, self.map, visable=False)
        self.env.new_episode()
        s = self.preprocess(self.env.get_state().screen_buffer)
        while True:

            a = self.model.choose_action(s)
            # env feeback
            self.env.set_action(self.action_dim[a])
            self.env.advance_action(frame_skip)
            done = self.env.is_episode_finished()
            #time.sleep(0.12)

            if not done:
                n_s = self.preprocess(self.env.get_state().screen_buffer)
                s = n_s
            else:
                print('enjoy result rewards : {}\n'.format(
                    self.env.get_total_reward()))
                break
        return self.env.get_total_reward()

    def run(self):
        plt_rewards = []
        a_loss_collects = []
        c_loss_collects = []
        loss_collects = []
        for episode in range(train_episodes):
            self.env.new_episode()
            s = self.preprocess(self.env.get_state().screen_buffer)
            done = False

            for t in range(0, horizon):
                if done:
                    self.env.new_episode()
                    s = self.preprocess(self.env.get_state().screen_buffer)

                a = self.model.choose_action(s)
                # env feeback
                r = self.env.make_action(self.action_dim[a], frame_skip)
                #self.env.advance_action(frame_skip)
                #r = self.env.get_last_reward()
                done = self.env.is_episode_finished()

                if not done:
                    n_s = self.preprocess(self.env.get_state().screen_buffer)
                    s = n_s
                else:
                    n_s = np.zeros(
                        [resolution[0], resolution[1], resolution_dim])

                self.rollout.append(s, a, r, n_s, done)

            if done:
                last_value = 0
            else:
                s = s[np.newaxis, :]
                last_value = self.model.get_v(s)

            returns = self.model.calculate_returns(
                self.rollout.r, self.rollout.done, last_value)
            self.rollout.r = returns[:-1]
            self.model.train(self.rollout)
            self.rollout.flush()

            #if episode % 20 == 0:
            
            #a_loss_collects.append(a_loss)
            #c_loss_collects.append(c_loss)
            #loss_collects.append(loss)
            #self.plot_loss(loss_collects, a_loss_collects, c_loss_collects)

            if episode % 200 == 0 and episode != 0:
                self.model.save_model(episode)
            if episode % 20 == 0 and episode != 0:
                print('Episode: {} '.format(episode))
                reward = self.enjoy()
                plt_rewards.append(reward)
                self.plot_durations(plt_rewards)
                #self.env = restart_doom(self.env, self.map, visable=False)


agent = Agent()
agent.run()
