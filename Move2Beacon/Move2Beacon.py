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
from dqn import DQN, get_copy_var_ops, replay_train
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
        replay_buffer = deque(maxlen=1000)
        sess = tf.Session()
        mainDQN = DQN(sess, 2, 4, name='main')
        targetDQN = DQN(sess, 2, 4, name='target')
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './Move2Beacon/model.cpkt')
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            while not 331 in obs[0].observation["available_actions"]:
                actions = actAgent2Pysc2(100,obs)
                obs = env.step(actions=[actions])
            state = obs2state(obs)
            print('episode start')
            global_step = 0
            random_rate = 0
            e = 1. / ((episodes / 10) + 1)
            reward = 0
            while not done: 
                global_step += 1
                time.sleep(0.05)
                if np.random.rand() < e:
                    random_rate += 1
                    action = random.randrange(4)
                else:
                    action = np.argmax(mainDQN.predict(state))
                #action = np.argmax(mainDQN.predict(state))
                actions = actAgent2Pysc2(action,obs)
                obs = env.step(actions=[actions])
                for i in range(1):
                    actions = actAgent2Pysc2(100,obs)
                    obs = env.step(actions=[actions])
                distance = obs2distance(obs)
                if global_step == 1:
                    pre_distance = distance
                next_state = obs2state(obs)
                reward = -(distance-pre_distance)*400
                #print(reward)
                if distance < 0.02 or global_step == 200:   # 게임 종료시
                    if distance < 0.02:
                        reward = 50
                    if global_step == 200:
                        reward = -10
                    done = True
                
                #print(next_state, reward)
                replay_buffer.append((state, action, reward, next_state, done))
                
                if distance < 0.02 or global_step == 200:   # 게임 종료시
                    if len(replay_buffer) > BATCH_SIZE:
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                        sess.run(copy_ops)
                        print('model trained')
                        saver.save(sess, './Move2Beacon/model.cpkt')
                    print(reward, episodes, random_rate/global_step)
                    break
                state = next_state
                pre_distance = distance

def test():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul) as env:
        sess = tf.Session()
        mainDQN = DQN(sess, 2, 4, name='main')
        targetDQN = DQN(sess, 2, 4, name='target')
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './Move2Beacon/model.cpkt')
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        for episodes in range(EPISODES):
            done = False
            obs = env.reset()
            while not 331 in obs[0].observation["available_actions"]:
                actions = actAgent2Pysc2(100,obs)
                obs = env.step(actions=[actions])
            state = obs2state(obs)
            print('episode start')
            global_step = 0
            random_rate = 0
            e = 1. / ((episodes / 10) + 1)
            reward = 0
            while not done: 
                time.sleep(0.13)
                global_step += 1

                action = np.argmax(mainDQN.predict(state))
                actions = actAgent2Pysc2(action,obs)
                obs = env.step(actions=[actions])
                for i in range(1):
                    actions = actAgent2Pysc2(100,obs)
                    obs = env.step(actions=[actions])
                distance = obs2distance(obs)
                if global_step == 1:
                    pre_distance = distance
                next_state = obs2state(obs)
                reward = -(distance-pre_distance)*400
                if distance < 0.02 or global_step == 200:   # 게임 종료시
                    if distance < 0.02:
                        reward = 50
                    if global_step == 200:
                        reward = -10
                    done = True
                
                if distance < 0.02 or global_step == 200:   # 게임 종료시
                    print(reward, episodes, random_rate/global_step)
                    break
                state = next_state
                pre_distance = distance


if __name__ == '__main__':
    #train()
    test()