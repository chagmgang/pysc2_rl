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
from A2C import Actor, Critic
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
EPISODES = 10000
BATCH_SIZE = 500
# main function, create env, define model, learn from observation and save model
def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul) as env:
        sess = tf.Session()
        actor = Actor(sess, n_features=2, n_actions=4, lr=0.001)
        critic = Critic(sess, n_features=2, lr=0.001)
        sess.run(tf.global_variables_initializer())
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            while not 331 in obs[0].observation["available_actions"]:
                actions = actAgent2Pysc2(100,obs)
                obs = env.step(actions=[actions])
            state = np.array(obs2state(obs))
            print('episode start')
            global_step = 0
            reward = 0
            while not done: 
                global_step += 1
                time.sleep(0.2)
                action = actor.choose_action(state)
                actions = actAgent2Pysc2(action,obs)
                obs = env.step(actions=[actions])
                for i in range(3):
                    actions = no_operation(obs)
                    obs = env.step(actions=[actions])
                distance = obs2distance(obs)
                if global_step == 1:
                    pre_distance = distance
                next_state = np.array(obs2state(obs))
                reward = -(distance-pre_distance)*400
                
                if distance < 0.03 or global_step == 200:   # 게임 종료시
                    if distance < 0.03:
                        reward = 10
                    if global_step == 200:
                        reward = -10
                    done = True
                
                td_error = critic.learn(state, reward, next_state)
                actor.learn(state, action, td_error)

                if distance < 0.03 or global_step == 200:   # 게임 종료시
                    break
                state = next_state
                pre_distance = distance

if __name__ == '__main__':
    train()