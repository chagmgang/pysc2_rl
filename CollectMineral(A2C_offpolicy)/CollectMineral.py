import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from action_group import actAgent2Pysc2, no_operation, gather
from state_group import obs2state, obs2done
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
feature_number = 64*64
# main function, create env, define model, learn from observation and save model
def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul) as env:
        sess = tf.Session()
        actor = Actor(sess, n_features=feature_number, n_actions=action_number, lr=0.001)
        critic = Critic(sess, n_features=feature_number, lr=0.001)
        sess.run(tf.global_variables_initializer())
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()            
            while not 331 in obs[0].observation["available_actions"]:
                actions = actAgent2Pysc2(100, obs)
                obs = env.step(actions=[actions])
            actions = gather(obs)
            obs = env.step(actions=[actions])
            for i in range(130):
                actions = no_operation(obs)
                obs = env.step(actions=[actions])
            state = obs2state(obs).reshape(64*64)
            end_step = 200
            global_step = 0
            print('episode start')
            while not done: 
                global_step += 1
                actions = no_operation(obs)
                obs = env.step(actions=[actions])
                next_state = obs2state(obs).reshape(64*64)
                reward = obs[0].reward
                done = obs2done(obs, global_step, end_step)

                if done:   # 게임 종료시
                    break
                state = next_state
if __name__ == '__main__':
    train()