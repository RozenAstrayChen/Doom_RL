import tensorflow as tf
import numpy as np
import sys
import itertools as it
from basic.env import *
from basic.config import *
from basic.process import *


class Agent(Process):
    def __init__(self, map=map_health):
        self.mode = 'train'
        self.map = map
        # init env
        self.env = init_doom(map, visable=False)
        
        n = self.env.get_available_buttons_size()
        self.action_dim = np.identity(n,dtype=int).tolist()
        self.action_num = n

        self._init_input()
        self._init_nn()
        self._init_op()

        self.a_buffer = []
        self.s_buffer = []
        self.r_buffer = []
        self.a_p_r_buffer = []
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(
                tf.float32, [None] + list(resolution) + [resolution_dim], name='s')
            self.a = tf.placeholder(
                tf.int32, [
                    None,
                ], name='a')
            self.r = tf.placeholder(
                tf.float32, [
                    None,
                ], name='r')
            self.adv = tf.placeholder(
                tf.float32, [
                    None,
                ], name='adv')
            self.a_p_r = tf.placeholder(
                tf.float32, [
                    None,
                ], name='a_p_r')

    def _init_nn(self, *args):
        self.advantage, self.value = self._init_critic_net('critic_net')
        self.a_prob_eval, self.a_logits_eval = self._init_actor_net(
            'eval_actor_net')
        self.a_prob_target, self.a_logits_target = self._init_actor_net(
            'target_actor_net', trainable=False)

    def _init_op(self):
        with tf.variable_scope('critic_loss_func'):
            self.c_loss_func = tf.losses.mean_squared_error(
                labels=self.r, predictions=self.value)
        with tf.variable_scope('critic_optimizer'):
            self.c_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.c_loss_func)
        with tf.variable_scope('update_target_actor_net'):
            # get eval w, b.
            params_e = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_actor_net')
            params_t = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
            self.update_target_a_op = [
                tf.assign(t, e) for t, e in zip(params_t, params_e)
            ]

        with tf.variable_scope('actor_loss_func'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                axis=1)

            new_policy_prob = tf.gather_nd(
                params=self.a_prob_eval, indices=a_indices)  # shape=(None, )

            old_policy_prob = tf.gather_nd(
                params=self.a_prob_target, indices=a_indices)  # shape=(None, )

            ratio = new_policy_prob / old_policy_prob

            surr = ratio * self.adv  # surrogate loss

            EPSILON = 0.2
            # clipped surrogate objective
            self.a_loss_func = -tf.reduce_mean(
                tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) *
                    self.adv))
        with tf.variable_scope('actor_optimizer'):
            self.a_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.a_loss_func)

    def _init_actor_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            '''
            84, 84 -> 20, 20
            '''
            conv1 = tf.layers.conv2d(
                self.s,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                ),
                name='conv1',
                trainable=trainable)
            '''
            9, 9 -> 3, 3
            '''
            conv2 = tf.layers.conv2d(
                conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                ),
                name='conv2',
                trainable=trainable)          
            conv2_flatten = tf.layers.flatten(conv2)
            

            f_dense = tf.layers.dense(
                conv2_flatten,
                512,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable,
                name='fc1')

            # action logits
            a_logits = tf.layers.dense(
                f_dense,
                self.action_num,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable)
            # action prob
            a_prob = tf.nn.softmax(a_logits)

            return a_prob, a_logits

    def _init_critic_net(self, scope):
        '''
        84, 84 -> 20, 20
        '''
        # first conv
        conv1 = tf.layers.conv2d(
            self.s,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
            ),
            name='conv1')
        '''
        20, 20 -> 9, 9
        '''
        conv2 = tf.layers.conv2d(
            conv1,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
            ),
            name='conv2')
        conv2_flatten = tf.layers.flatten(conv2)
        

        f_dense = tf.layers.dense(
            conv2_flatten,
            512,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='fc1')

        value = tf.layers.dense(
            f_dense,
            1,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='critic')

        value = tf.reshape(value, [
            -1,
        ])

        # advantage
        advantage = self.r - value
        return advantage, value

    def predict(self, s):
        # calculate a eval prob
        a_prob_eval, a_prob_target = self.session.run(
            [self.a_prob_eval, self.a_prob_target], {self.s: [s]})
        
        #a_p_r = np.max(a_prob_eval) / np.max(a_prob_target)
        #self.a_p_r_buffer.append(a_p_r)
        action =  np.random.choice(
            range(a_prob_eval.shape[1]), p=a_prob_eval.ravel())
        # 
        a_p_r = a_prob_eval[0][action] / a_prob_target[0][action]
        
        self.a_p_r_buffer.append(a_p_r)

        return action

    def snapshot(self, s, a, r, _):
        self.a_buffer.append(a)
        self.s_buffer.append(s)
        self.r_buffer.append(r)

    def train(self):
        self.session.run(self.update_target_a_op)
        # copy r_buffer
        r_buffer = self.r_buffer
        # Init r_tau
        r_tau = 0
        # Calcuate r_tau
        for index in reversed(range(0, len(r_buffer))):
            r_tau = r_tau * gamma + r_buffer[index]
            self.r_buffer[index] = r_tau
        # Calcuate advantage
        adv_buffer = self.session.run(self.advantage, {
            self.s: self.s_buffer,
            self.r: self.r_buffer
        })
        # minimize loss
        [
            self.session.run(
                [self.a_optimizer, self.c_optimizer], {
                    self.adv: adv_buffer,
                    self.s: self.s_buffer,
                    self.a: self.a_buffer,
                    self.r: self.r_buffer,
                    self.a_p_r: self.a_p_r_buffer,
                }) for _ in range(5)
        ]
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []
        self.a_p_r_buffer = []

    def save_model(self, episode):
        path = './temp/' + str(episode)
        name = '/model'
        save_path = path + name
        self.saver.save(self.session, save_path)

    def load_model(self, episode):
        path = './temp/' + str(episode)
        name = '/model'
        save_path = path + name
        self.saver.restore(self.session, save_path)

    def run(self):
        # stack frames
        plt_rewards = []
        render_flag = True
        if self.mode == 'train':
            for episode in range(train_episodes):
                self.env.new_episode()
                s = self.preprocess2(self.env.get_state().screen_buffer)

                while True:
                    if episode > 4000 and render_flag == False:
                        self.env = init_doom(scenarios=self.map, visable=True)
                        render_flag = True
                    a = self.predict(s)
                    # enviroment feeback
                    self.env.set_action(self.action_dim[a])
                    self.env.advance_action(frame_skip+1)
                    r = self.env.get_last_reward()
                    done = self.env.is_episode_finished()

                    if done:
                        # next state doesn't saved
                        s_n = np.zeros((resolution[0], resolution[1], resolution_dim), dtype=np.int)
                        self.snapshot(s, a, r, s_n)
                        break
                    else:
                        s_n = self.preprocess2(
                            self.env.get_state().screen_buffer)
                        self.snapshot(s, a, r, s_n)
                        s = s_n

                self.train()
                if episode % 20 == 0:
                    print('Episode: {} | Rewards: {}'.format(
                        episode, self.env.get_total_reward()))
                    plt_rewards.append(self.env.get_total_reward())
                    self.plot_durations(plt_rewards)
                if episode % 200 == 0 and episode != 0:
                    print('save model!')
                    self.save_model(episode)
            self.plot_save(plt_rewards)
        else:
            import time
            self.load_model(1000)
            self.env = init_doom(scenarios=self.map, visable=True)
            for episode in range(watch_episodes):
                self.env.new_episode()
                s = self.preprocess2(self.env.get_state().screen_buffer)
                #s = self.frames_reshape(s)
                while True:
                    a = self.predict(s)
                    self.env.set_action(self.action_dim[a])
                    self.env.advance_action(frame_skip+1)
                    done = self.env.is_episode_finished()
                    time.sleep(0.12)
                    if done:
                        print('get total rewards = ',
                              self.env.get_total_reward())
                        break
                    else:
                        s_n = self.preprocess2(
                            self.env.get_state().screen_buffer)
                        #s_n = self.frames_reshape(s_n)
                        s = s_n


ag = Agent()
#ag.mode = "not"
ag.run()