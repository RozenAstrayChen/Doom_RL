import numpy as np
import tensorflow as tf
from basic.config import *


class Actor(object):
    def __init__(self, sess, dim_state, n_actions, lr=actor_lr):
        self.sess = sess

        self.s = tf.placeholder(
            tf.float32, [None] + list(resolution) + [dim_state], 'state')
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None,
                                       "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            conv1 = tf.layers.conv2d(
                self.s,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.relu,
                padding='valid')
            conv2 = tf.layers.conv2d(
                conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.relu,
                padding='valid')

            conv2_flat = tf.layers.flatten(conv2)

            f_dense = tf.layers.dense(
                conv2_flat,
                units=128,
                activation=tf.nn.relu)

            self.acts_prob = tf.layers.dense(
                f_dense,
                units=n_actions,
                activation=tf.nn.softmax,
                #kernel_initializer=tf.random_normal_initializer(0., .01),
                bias_initializer=None)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]

        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob,
                              {self.s: s})  # get probabilities for all actions
        return np.random.choice(
            np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, dim_state, n_actions, lr=critic_lr):
        self.sess = sess

        self.s = tf.placeholder(
            tf.float32, [None] + list(resolution) + [dim_state], 'state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            
            conv1 = tf.layers.conv2d(
                self.s,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.relu,
                padding='valid')
            conv2 = tf.layers.conv2d(
                conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.relu,
                padding='valid')

            conv2_flat = tf.layers.flatten(conv2)

            f_dense = tf.layers.dense(
                conv2_flat,
                units=128,
                activation=tf.nn.relu)

            self.v = tf.layers.dense(
                f_dense,
                units=1,
                activation=None,
                #kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=None)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            # improve 
            #self.td_error = 0.5 * self.td_error
            self.loss = tf.square(
                self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {
            self.s: s,
            self.v_: v_,
            self.r: r
        })
        return td_error
