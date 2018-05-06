import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from action_group import actAgent2Pysc2, no_operation
from state_group import obs2state, obs2distance
import numpy as np
import random
import tensorflow as tf
from collections import deque
import time
import math
from a2c import Actor, Critic, discount_rewards
# Define the constant
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index
friendly = 1
neutral = 3
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP           = actions.FUNCTIONS.no_op.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ALL  = [0]
_NOT_QUEUED  = [0]
step_mul = 4
FLAGS = flags.FLAGS
EPISODES = 200
BATCH_SIZE = 500
action_number = 4
feature_number = 2
# main function, create env, define model, learn from observation and save model
def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul) as env:
        sess = tf.Session()
        actor = Actor(sess, n_features=feature_number, n_actions=action_number, lr=0.001)
        critic = Critic(sess, n_features=feature_number, lr=0.001)
        #spend_time = tf.placeholder(tf.float32)
        #rr = tf.summary.scalar('reward', spend_time)
        #writer = tf.summary.FileWriter('./Move2Beacon(A2C_offpolicy)/board/off_a2c', sess.graph)
        saver = tf.train.Saver()
        #merged = tf.summary.merge_all()
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            state = np.array(obs2state(obs))
            global_step = 0
            reward = 0
            end_distance = 0.02
            end_step = 100

            states = np.empty(shape=[0,feature_number])
            next_states = np.empty(shape=[0,feature_number])
            rewards = np.empty(shape=[0,1])
            actions_list = np.empty(shape=[0,action_number])
            
            while not done: 
                global_step += 1
                time.sleep(0.05)
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                predict_actor = actor.choose_action(state)
                actions = actAgent2Pysc2(predict_actor,obs)
                obs = env.step(actions=[actions])
                for i in range(1):
                    actions = no_operation(obs)
                    obs = env.step(actions=[actions])
                distance = obs2distance(obs)
                if global_step == 1:
                    pre_distance = distance
                next_state = np.array(obs2state(obs))
                reward = -np.sign(distance-pre_distance)
                if reward == 0:
                    reward = -1
                reward = 0
                if distance < end_distance or global_step == end_step:   # 게임 종료시
                    if distance < end_distance:
                        reward = 100
                    if global_step == end_step:
                        reward = -100
                    done = True
                
                states = np.vstack([states, state])
                next_states = np.vstack([next_states, next_state])
                rewards = np.vstack([rewards, reward])
                action = np.zeros(action_number)
                action[predict_actor] = 1
                actions_list = np.vstack([actions_list, action])

                if distance < end_distance or global_step == end_step:   # 게임 종료시
                    if distance < end_distance:
                        print('success', global_step, episodes)
                    else:
                        print('failed', global_step, episodes)
                    discouned_rewards = discount_rewards(rewards)
                    td_error = critic.learn(states, discouned_rewards, next_states)
                    actor.learn(states, actions_list, td_error)
                    #summary = sess.run(merged, feed_dict={spend_time: global_step})
                    #writer.add_summary(summary, episodes)
                    #saver.save(sess, './Move2Beacon(A2C_offpolicy)/model/model.ckpt')
                    break
                state = next_state
                pre_distance = distance

if __name__ == '__main__':
    train()