import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
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
    with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul,
                        screen_size_px=(32, 32), minimap_size_px=(32, 32)) as env:
        Policy = Policy_net('policy', 32*32*2, 4)
        Old_Policy = Policy_net('old_policy', 32*32*2, 4)
        PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('a')
            saver.restore(sess, './model/model.ckpt')
            print('a')
            #sess.run(tf.global_variables_initializer())
            for episodes in range(EPISODES):
                done = False
                obs = env.reset()
                while not 331 in obs[0].observation["available_actions"]:
                    actions = actAgent2Pysc2(100, obs)
                    obs = env.step(actions=[actions])
                actions = gather(obs)
                obs = env.step(actions=[actions])
                end_step = 200
                global_step = 0
                score = 0
                reward = 0
                for i in range(100):
                    time.sleep(0.01)
                    actions = no_operation(obs)
                    obs = env.step(actions=[actions])       
                state = obs2state(obs)
                observations = []
                actions_list = []
                v_preds = []
                rewards = []

                print('episode start')
                while not done:
                    global_step += 1
                    time.sleep(0.05)
                    state = np.stack([state]).astype(dtype=np.float32)
                    act, v_pred = Policy.act(obs=state, stochastic=True)
                    act, v_pred = np.asscalar(act), np.asscalar(v_pred)
                    actions = actAgent2Pysc2(act, obs)
                    #while not 331 in obs[0].observation["available_actions"]:
                    #    actions = actAgent2Pysc2(100, obs)
                    #    obs = env.step(actions=[actions])
                    obs = env.step(actions=[actions])
                    
                    if global_step == end_step or obs2done(obs) >= 1900 :    # 게임 time을 다 사용하거나 미네랄을 다 먹었을 경우 게임이 끝난다.
                        done = True
                    
                    next_state = obs2state(obs)
                    reward = obs[0].reward

                    if reward == 0:
                        reward = -0.1

                    if done:
                        if obs2done(obs) >= 1900:   # 게임이 종료되었는데 미네랄을 다 먹었으면
                            reward = 3
                        else:                       # 게임이 종료되었는데 미네랄을 다 못먹으면
                            reward = -3   

                    score += reward

                    observations.append(state)
                    actions_list.append(act)
                    v_preds.append(v_pred)
                    rewards.append(reward)

                    if done:   # 게임 종료시
                        v_preds_next = v_preds[1:] + [0]
                        gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                        observations = np.reshape(observations, newshape=[-1, 32*32*2])
                        actions = np.array(actions_list).astype(dtype=np.int32)
                        rewards = np.array(rewards).astype(np.float32)
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                        gaes = np.array(gaes).astype(dtype=np.float32)
                        gaes = (gaes - gaes.mean())
                        PPO.assign_policy_parameters()
                        inp = [observations, actions, rewards, v_preds_next, gaes]
                        for epoch in range(5):
                            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)
                            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                            PPO.train(obs=sampled_inp[0],
                                    actions=sampled_inp[1],
                                    rewards=sampled_inp[2],
                                    v_preds_next=sampled_inp[3],
                                    gaes=sampled_inp[4])
                        print(episodes, score)
                        save_path = saver.save(sess, './model/model.ckpt')
                        if episodes == 0:
                            f = open('test2.csv', 'w', encoding='utf-8', newline='')
                        else:
                            f = open('test2.csv', 'a', encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        wr.writerow([episodes, score])
                        f.close()
                        break
                    state = next_state
if __name__ == '__main__':
    train()