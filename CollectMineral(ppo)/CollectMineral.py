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
from policy_net import Policy_net
from ppo import PPOTrain
import csv
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
EPISODES = 20000000
BATCH_SIZE = 500
action_number = 4
feature_number = 64*64
# main function, create env, define model, learn from observation and save model
def train():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul) as env:
        Policy = Policy_net('policy')
        Old_Policy = Policy_net('old_policy')
        PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for episodes in range(EPISODES):
                done = False
                obs = env.reset()          
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                state = obs2state(obs)
                end_step = 300
                global_step = 0
                score = 0
                reward = 0
                mineral = 0 
                observations = []
                actions_list = []
                v_preds = []
                rewards = []
                pre_position_marine_y, pre_position_marine_x = (obs[0].observation["screen"][_SELECTED_UNIT] == True).nonzero()
                
                print('episode start')
                while not done:
                    global_step += 1
                    time.sleep(0.03)
                    state = np.stack([state]).astype(dtype=np.float32)
                    act, v_pred = Policy.act(obs=state, stochastic=True)
                    act, v_pred = np.asscalar(act), np.asscalar(v_pred)
                    actions = actAgent2Pysc2(act, obs)
                    obs = env.step(actions=[actions])
                    
                    next_state = obs2state(obs)
                    reward = obs[0].reward
                    mineral += reward

                    if reward == 0:
                        reward = -0.1

                    score += reward

                    done = obs2done(obs, global_step, end_step)
                    position_marine_y, position_marine_x = (obs[0].observation["screen"][_SELECTED_UNIT] == True).nonzero()

                    if done:
                        if global_step == end_step:
                            reward = -3
                        else:
                            reward = 3
                    
                    if list(pre_position_marine_y) == list(position_marine_y):
                        if list(pre_position_marine_x) == list(position_marine_x):
                            reward += -0.1
                    
                    observations.append(state)
                    actions_list.append(act)
                    v_preds.append(v_pred)
                    rewards.append(reward)

                    if done:   # 게임 종료시
                        v_preds_next = v_preds[1:] + [0]
                        gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                        observations = np.reshape(observations, newshape=[-1, 64*64*2])
                        actions_list = np.array(actions_list).astype(dtype=np.int32)
                        rewards = np.array(rewards).astype(dtype=np.float32)
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                        gaes = np.array(gaes).astype(dtype=np.float32)
                        gaes = (gaes - gaes.mean())

                        PPO.assign_policy_parameters()

                        inp = [observations, actions_list, rewards, v_preds_next, gaes]
                        for epoch in range(5):
                            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                            PPO.train(obs=sampled_inp[0], actions=sampled_inp[1],
                                        rewards=sampled_inp[2], v_preds_next=sampled_inp[3],
                                        gaes=sampled_inp[4])
                        if episodes == 0:
                            f = open('output.csv', 'w', encoding='utf-8', newline='')
                        else:
                            f = open('output.csv', 'a', encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        wr.writerow([episodes, score, mineral])
                        f.close()
                        print(episodes, global_step, score, mineral)
                        break
                    state = next_state
                    pre_position_marine_y, pre_position_marine_x = position_marine_y, position_marine_x
if __name__ == '__main__':
    train()