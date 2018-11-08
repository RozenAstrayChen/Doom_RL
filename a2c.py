import numpy as np
import tensorflow as tf
import numpy as np
import sys
import itertools as it
from basic.env import *
from basic.config import *
from basic.process import *
from net.a2c_net import Actor, Critic


class A2C(Process):
    def __init__(self, map=map_health, output_graph=False):
        self.mode = 'train'
        self.map = map
        #init env
        self.env = init_doom(map, visable=False)
        
        
        n = self.env.get_available_buttons_size()
        self.action_dim = np.identity(n ,dtype=int).tolist()
        self.action_num = n
        self.state_dim = resolution_dim  # RGB

        self.sess = tf.Session()
       

        self._init_neural()
        self._init_graph(output_graph)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def _init_neural(self):
        self.actor = Actor(
            self.sess,
            self.state_dim,
            self.action_num,
            lr=actor_lr)

        self.critic = Critic(
            self.sess,
            self.state_dim,
            self.action_num,
            lr=critic_lr)
    

    def _init_graph(self, graph):
        if graph:
            tf.summary.FileWriter('result/a2c/logs/', self.sess.graph)
    
    def save_model(self):
        path = './result/a2c/model'  #+ str(episode)
        name = '/a2c'
        save_path = path + name
        self.saver.save(self.sess, save_path)

    def load_model(self):
        path = './result/a2c/model'  #+ str(episode)
        name = '/a2c'
        save_path = path + name
        self.saver.restore(self.sess, save_path)

    def train(self):
        collect_rewards = []
        for episode in range(train_episodes):
            self.env.new_episode()
            s = self.preprocess(self.env.get_state().screen_buffer)

            while True:
                
                a = self.actor.choose_action(s)
                #r = self.env.make_action(self.action_dim[a], frame_repeat)
                self.env.set_action(self.action_dim[a])
                self.env.advance_action(frame_skip)
                r = self.env.get_last_reward()
                
                done = self.env.is_episode_finished()

                if done:
                    s_ = np.zeros(
                        (resolution[0], resolution[1], 3), dtype=np.int)
                else:
                    s_ = self.preprocess(self.env.get_state().screen_buffer)

                td_error = self.critic.learn(s, r, s_)
                self.actor.learn(s, a, td_error)

                s = s_

                if done:
                    break  # game over

            if episode % 20 == 0:
                print('Episode: {} | Rewards: {}'.format(
                    episode, self.env.get_total_reward()))
                collect_rewards.append(self.env.get_total_reward())
                self.plot_durations(collect_rewards)
            if episode % 100 == 0 and episode != 0:
                print('save model')
                self.save_model()
    
    def enjoy(self):
        import time
        self.load_model()
        self.env = init_doom(self.map, visable=True)
        for episode in range(enjoy_episodes):
            self.env.new_episode()
            s = self.preprocess(self.env.get_state().screen_buffer)

            while True:
                
                a = self.actor.choose_action(s)
                r = self.env.make_action(self.action_dim[a], frame_repeat)
                #self.env.set_action(self.action_dim[a])
                #self.env.advance_action(frame_skip)
                #r = self.env.get_last_reward()
                
                done = self.env.is_episode_finished()

                if done:
                    s_ = np.zeros(
                        (resolution[0], resolution[1], resolution_dim), dtype=np.int)
                else:
                    s_ = self.preprocess(self.env.get_state().screen_buffer)

                time.sleep(0.028)

                s = s_

                if done:
                    break  # game over

            
            print('Episode: {} | Rewards: {}'.format(
                episode, self.env.get_total_reward()))


agent = A2C()
#agent.enjoy()
agent.train()