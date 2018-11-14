import tensorflow as tf
import numpy as np
import sys
import itertools as it
from basic.env import *
from basic.config import *
from basic.process import *
from basic.rollout import *
from ppo_batch import *

class Agent(Process):
    def __init__(self, map=map_health):
        self.map = map
        self.env = init_doom(map, visable=False)
        self.sess = tf.Session()

        n = self.env.get_available_buttons_size()
        self.action_dim = np.identity(n,dtype=int).tolist()
        self.action_num = n

        self.model = PPO_batch(self.sess, self.action_num)
        self.rollout = Rollout()

    def run(self):
        plt_rewards = []
        for episode in range(train_episodes):
            self.env.new_episode()
            s = self.preprocess(self.env.get_state().screen_buffer)
            done = False
            while True:
                
                a = self.model.choose_action(s)
                # env feeback
                self.env.set_action(self.action_dim[a])
                self.env.advance_action(frame_skip)
                r = self.env.get_last_reward()
                done = self.env.is_episode_finished()

                self.rollout.append(s, a, r)

                if not done:
                    n_s = self.preprocess(self.env.get_state().screen_buffer)
                    s = n_s
                
                else:
                    r_tau = 0  
                    discount_r = []
                    for r in self.rollout.r[::-1]:
                        r_tau = r_tau * gamma + r
                        discount_r.append(r_tau)
                    discount_r.reverse()
                    # update
                    self.rollout.r = discount_r
                    s_batch, a_batch, r_batch, = self.rollout.sample()
                    self.model.train(s_batch, a_batch, r_batch)
                    self.rollout.clean()
                    break

            
            if episode % 20 == 0:
                    print('Episode: {} | Rewards: {}'.format(
                        episode, self.env.get_total_reward()))
                    plt_rewards.append(self.env.get_total_reward())
                    self.plot_durations(plt_rewards)
                

agent = Agent()
agent.run()





