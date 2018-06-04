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
from policy_net import Policy_net
from ppo import PPOTrain

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
    with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul,
                        screen_size_px=(16, 16), minimap_size_px=(16, 16)) as env:
        Policy = Policy_net('policy', 16*16*2, 4)
        Old_Policy = Policy_net('old_policy', 16*16*2, 4)
        PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
        with tf.Session() as sess:
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

                observations = []
                actions_list = []
                v_preds = []
                rewards = []

                while not done: 
                    global_step += 1
                    time.sleep(0.05)

                    state = np.stack([state]).astype(dtype=np.float32)
                    act, v_pred = Policy.act(obs=state, stochastic=True)
                    act, v_pred = np.asscalar(act), np.asscalar(v_pred)
                    actions = actAgent2Pysc2(act, obs)
                    obs = env.step(actions=[actions])

                    for i in range(1):
                        actions = no_operation(obs)
                        obs = env.step(actions=[actions])
                    distance = obs2distance(obs)
                    if global_step == 1:
                        pre_distance = distance
                    next_state = np.array(obs2state(obs))
                    reward = -10*(distance-pre_distance)
                    #if reward < 0 :
                    #    reward = -0.01
                    #if reward <= 0:
                    #    reward = 0
                    #elif reward > 0:
                    #    reward = 0
                    reward = -0.01
                    if distance < 0.03 or global_step == 100:   # 게임 종료시
                        if distance < 0.03:
                            reward = 1
                        if global_step == 200:
                            reward = -1
                        done = True
                    
                    observations.append(state)
                    actions_list.append(act)
                    v_preds.append(v_pred)
                    rewards.append(reward)

                    if distance < 0.03 or global_step == 100:   # 게임 종료시
                        v_preds_next = v_preds[1:] + [0]
                        gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                        observations = np.reshape(observations, newshape=[-1, 16*16*2])
                        actions = np.array(actions_list).astype(dtype=np.int32)
                        rewards = np.array(rewards).astype(dtype=np.float32)
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                        gaes = np.array(gaes).astype(dtype=np.float32)
                        gaes = (gaes - gaes.mean())

                        PPO.assign_policy_parameters()

                        inp = [observations, actions, rewards, v_preds_next, gaes]
                        for epoch in range(5):
                            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                            PPO.train(obs=sampled_inp[0],
                                    actions=sampled_inp[1],
                                    rewards=sampled_inp[2],
                                    v_preds_next=sampled_inp[3],
                                    gaes=sampled_inp[4])
                        print(episodes, global_step)
                        break
                    state = next_state
                    pre_distance = distance

if __name__ == '__main__':
    train()