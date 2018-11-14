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

        self.sess.run(tf.global_variables_initializer())

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
                tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            params_old = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            self.update_params = [
                tf.assign(old, now) for old, now in zip(params_old, params)
            ]
        
        with tf.variable_scope('advantage'):
            self.adv = self.r_in - self.v_eval
        
        with tf.variable_scope('loss'):
            # Actor Loss
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a_in)[0], dtype=tf.int32), self.a_in],
                axis=1)
            # theta
            pi_prob = tf.gather_nd(params=self.a_prob_eval, indices=a_indices)

            # old_theta
            pi_old_prob = tf.gather_nd(
                params=self.a_prob_targ, indices=a_indices)

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
            self.entropy = tf.reduce_sum(
                self.a_prob_eval * tf.log(self.a_prob_eval))  # encourage exploration

            self.loss = self.a_loss + (coeff_value * self.c_loss) - (
                self.entropy * coeff_entropy)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.loss)


    def _init_net(self):
        self.v_eval, self.a_prob_eval = self._init_ac_net('eval_net')
        self.v_targ, self.a_prob_targ = self._init_ac_net('target_net', trainable=False)

    def _init_ac_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            conv1 = tf.layers.conv2d(
                self.s_in,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                trainable=trainable)

            conv2 = tf.layers.conv2d(
                conv1,
                filters=64,
                kernel_size=[8,8],
                strides=[4,4],
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                trainable=trainable
            )
            conv2_flatten = tf.layers.flatten(conv2)

            '''
            conv3 = tf.layers.conv2d(
                conv2,
                filters=64,
                kernel_size=[3, 3],
                strides=[1, 1],
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                trainable=trainable)
            conv3_flatten = tf.layers.flatten(conv3)
            '''
            f_dense = tf.layers.dense(
                conv2_flatten,
                512,
                tf.nn.relu,
                kernel_initializer=w_initializer,
                trainable=trainable)
            
            value = tf.layers.dense(
                f_dense,
                1,
                activation=None,
                kernel_initializer=w_initializer,
                trainable=trainable
            )
            action = tf.layers.dense(
                f_dense,
                self.action_n,
                activation=tf.nn.softmax,
                kernel_initializer=w_initializer,
                trainable=trainable
            )

            return value, action

    def get_v(self, s):
        v = self.sess.run([self.v_eval], feed_dict={self.s_in: [s]})
        return v[0][0]

    def choose_action(self, s):
        a_prob = self.sess.run(
            self.a_prob_targ, feed_dict={self.s_in: [s]})
        a = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return a
    
    def train(self, s, a, r):
        
        #update old theta
        self.sess.run(self.update_params)
        #newaxis
        r = r[:,np.newaxis]

        
        adv = self.sess.run(self.adv, {self.s_in: s, self.r_in: r})
        #adv = np.array(adv)
        adv = adv.ravel()
        #print('adv shape ', adv.shape)

        dict = {self.s_in: s, self.r_in: r, self.a_in: a, self.adv_in: adv}
        '''
        for _ in range(5):
            _, loss, a_loss, c_loss, entropy = self.sess.run(
                [
                    self.optimizer, self.loss, self.a_loss, self.c_loss,
                    self.entropy
                ],
                feed_dict=dict)

        self.rollout.clean()

        return a_loss, c_loss, entropy
        '''
        
        self.sess.run([self.optimizer], feed_dict=dict)

    
