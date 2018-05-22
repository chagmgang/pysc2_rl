import gym
import numpy as np
import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, state_size, action_size, temp=0.1):

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.tanh)
                layer_4 = tf.layers.dense(inputs=layer_3, units=action_size, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_4, temp), units=action_size, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=30, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            #print(tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs}))
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)