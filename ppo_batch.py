import tensorflow as tf
import numpy as np
import sys
import itertools as it
from basic.env import *
from basic.config import *
from basic.process import *
from basic.rollout import *


class PPO_batch():
    def __init__(self, sess, action_n):
        self.sess = sess
        #self.saver = tf.train.Saver()
        self.action_n = action_n

        self._init_input()
        self._init_net()
        self._init_op()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def save_model(self, episode):
        path = './temp/' + str(episode)
        name = '/model'
        save_path = path + name
        self.saver.save(self.sess, save_path)

    def load_model(self, episode):
        path = './temp/' + str(episode)
        name = '/model'
        save_path = path + name
        self.saver.restore(self.sess, save_path)

    def _init_input(self):
        with tf.variable_scope('input'):
            self.s_in = tf.placeholder(
                tf.float32, [None] + list(resolution) + [resolution_dim],
                name='s_in')
            self.a_in = tf.placeholder(tf.int32, [None], name='a_in')
            self.r_in = tf.placeholder(tf.float32, [None, 1], name='r_in')
            self.adv_in = tf.placeholder(tf.float32, [None], name='adv_in')

    def _init_op(self):
        with tf.variable_scope('update_params'):
            params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='theta')
            params_old = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='theta_old')
            self.update_params = [
                tf.assign(old, now) for old, now in zip(params_old, params)
            ]
        '''
        with tf.variable_scope('advantage'):
            self.adv = self.r_in - self.v_eval
        '''

        with tf.variable_scope('loss'):
            # Actor Loss
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a_in)[0], dtype=tf.int32), self.a_in],
                axis=1)
            # theta
            pi_prob = tf.gather_nd(params=self.a_prob, indices=a_indices)

            # old_theta
            pi_old_prob = tf.gather_nd(
                params=self.a_prob_old, indices=a_indices)

            # surrogate 1
            ratio = pi_prob / pi_old_prob

            surr1 = ratio * self.adv_in

            # surrogate 2
            surr2 = tf.clip_by_value(ratio, 1. - epsilon,
                                     1. + epsilon) * self.adv_in

            # estimate
            self.a_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            # Critic Loss
            self.c_loss = tf.reduce_mean(tf.square(self.adv_in))
            # dist entropy
            self.entropy = tf.reduce_mean(
                self.a_prob * tf.log(self.a_prob))  # encourage exploration

            self.loss = self.a_loss + self.c_loss - self.entropy * coeff_entropy

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.loss)

    def _init_net(self):
        self.v, self.a_prob = self._init_ac_net('theta')
        self.v_old, self.a_prob_old = self._init_ac_net(
            'theta_old', trainable=False)

    def _init_ac_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            '''
            84, 84 -> 20, 20
            '''
            conv1 = tf.layers.conv2d(
                self.s_in,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                ),
                trainable=trainable,
                padding='valid',
                name='conv1')
            conv1_batchnorm = tf.layers.batch_normalization(
                conv1, training=trainable, epsilon=1e-5, name='batch_norm1')

            conv1_out = tf.nn.elu(conv1_batchnorm, name="conv1_out")
            '''
            20, 20 -> 9, 9
            '''
            conv2 = tf.layers.conv2d(
                conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                ),
                trainable=trainable,
                padding='valid',
                name='conv2')
            conv2_batchnorm = tf.layers.batch_normalization(
                conv2, training=trainable, epsilon=1e-5, name='batch_norm2')

            conv2_out = tf.nn.elu(conv2_batchnorm, name="conv2_out")
            '''
            9, 9 -> 3, 3
            '''
            conv3 = tf.layers.conv2d(
                conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                ),
                trainable=trainable,
                padding='valid',
                name='conv3')
            conv3_batchnorm = tf.layers.batch_normalization(
                conv3, training=trainable, epsilon=1e-5, name='batch_norm3')

            conv3_out = tf.nn.elu(conv3_batchnorm, name="conv3_out")

            conv3_flatten = tf.layers.flatten(conv3_out)
            f_dense = tf.layers.dense(
                conv3_flatten,
                512,
                tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable,
                name='fc1')

            value = tf.layers.dense(
                f_dense,
                1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable,
                name='critic')

            action = tf.layers.dense(
                f_dense,
                self.action_n,
                activation=tf.nn.softmax,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                trainable=trainable,
                name='actor')

            return value, action

    def get_v(self, s):
        v = self.sess.run([self.v], feed_dict={self.s_in: s})
        return v[0][0]

    def choose_action(self, s):
        a_prob = self.sess.run(self.a_prob, feed_dict={self.s_in: [s]})
        a = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return a

    def calculate_returns(self, rewards, dones, last_value, gamma=0.99):
        rewards = np.array(rewards)
        dones = np.array(dones)
        # create nparray
        returns = np.zeros(rewards.shape[0] + 1)
        returns[-1] = last_value
        dones = 1 - dones
        for i in reversed(range(rewards.shape[0])):
            returns[i] = gamma * returns[i + 1] * dones[i] + rewards[i]

        return returns

    def train(self, memory):
        # First, update old theta
        self.sess.run(self.update_params)
        # Second calculate advantage
        s = np.array(memory.s)
        returns = np.array(memory.r)
        #newaxis
        returns = returns[:, np.newaxis]
        predicts = self.get_v(s)
        #print('ret', returns.shape, '\tand one is', returns[0])
        #print('predicts', predicts.shape, '\tand one is', predicts[0])
        adv = returns - predicts
        adv = adv.ravel()
        adv = (adv - adv.mean()) / adv.std()

        memory.adv_replace(adv)
        # update N times
        for _ in range(nupdates):
            # sample data
            s, a, adv = memory.sample()
            adv = adv.ravel()

            dict = {self.s_in: s, self.a_in: a, self.adv_in: adv}

            _, loss, a_loss, c_loss, entropy = self.sess.run(
                [
                    self.optimizer, self.loss, self.a_loss, self.c_loss,
                    self.entropy
                ],
                feed_dict=dict)

            #self.sess.run([self.optimizer], feed_dict=dict)
