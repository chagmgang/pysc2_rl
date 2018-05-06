import numpy as np
import tensorflow as tf
from collections import deque
import gym

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.float32, [None, n_actions], "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu)
            l1 = tf.layers.dense(inputs=l1, units=20, activation=tf.nn.relu)
            l1 = tf.layers.dense(inputs=l1, units=20, activation=tf.nn.relu)
            self.acts_prob = tf.layers.dense(l1, n_actions, activation=tf.nn.softmax)

        with tf.variable_scope('exp_v'):
            log_lik = -self.a * tf.log(self.acts_prob)
            log_lik_adv = log_lik * self.td_error
            self.exp_v = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.tanh)
            l1 = tf.layers.dense(inputs=l1, units=20, activation=tf.nn.tanh)
            l1 = tf.layers.dense(inputs=l1, units=20, activation=tf.nn.tanh)
            W = tf.Variable(tf.random_normal([20, 1]))
            self.v = tf.matmul(l1, W)

        with tf.variable_scope('squared_TD_error'):
            GAMMA = 0.99
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r