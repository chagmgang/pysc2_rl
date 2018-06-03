import gym
import numpy as np
import tensorflow as tf



class Policy_net:
    def __init__(self, name: str, temp=0.1):

        state_size = 64*64*2
        action_size = 4
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = None
        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name='obs')

            with tf.variable_scope('policy_net'):
                reshape = tf.reshape(self.obs, [-1, 64, 64, 2])
                layer_1 = tf.layers.conv2d(reshape, filters=4, kernel_size=[7, 7], activation=None, kernel_initializer=initializer)
                layer_2 = tf.layers.conv2d(layer_1, filters=16, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_3 = tf.layers.conv2d(layer_2, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_4 = tf.layers.conv2d(layer_3, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_5 = tf.layers.conv2d(layer_4, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_6 = tf.layers.dense(layer_5, 2, activation=tf.nn.relu, kernel_initializer=initializer)
                layer_7 = tf.reshape(layer_6, [-1, 34*34*2])
                layer_8 = tf.layers.dense(inputs=layer_7, units=20*20, activation=tf.nn.relu, kernel_initializer=initializer)
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_8, temp), units=action_size, activation=tf.nn.softmax)
                
            with tf.variable_scope('value_net'):
                reshape = tf.reshape(self.obs, [-1, 64, 64, 2])
                layer_1 = tf.layers.conv2d(reshape, filters=4, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_2 = tf.layers.conv2d(layer_1, filters=16, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_3 = tf.layers.conv2d(layer_2, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_4 = tf.layers.conv2d(layer_3, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_5 = tf.layers.conv2d(layer_4, filters=64, kernel_size=[7, 7], activation=tf.nn.relu, kernel_initializer=initializer)
                layer_6 = tf.layers.dense(layer_5, 2, activation=tf.nn.relu, kernel_initializer=initializer)
                layer_7 = tf.reshape(layer_6, [-1, 34*34*2])
                layer_8 = tf.layers.dense(inputs=layer_7, units=20*20, activation=None, kernel_initializer=initializer)
                self.v_preds = tf.layers.dense(inputs=layer_8, units=1, activation=None, kernel_initializer=initializer)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
           # print(tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs}))
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)