import sys
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# Define the constant
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
friendly = 1
neutral = 3
_SELECTED_UNIT = features.SCREEN_FEATURES.selected.index

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP           = actions.FUNCTIONS.no_op.id
_RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id


_SELECT_ALL  = [0]
_NOT_QUEUED  = [0]

def gather(obs):
    marine_y, marine_x = (obs[0].observation["screen"][_PLAYER_RELATIVE] == friendly).nonzero()
    marine_x, marine_y = np.mean(marine_x), np.mean(marine_y)
    action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x, marine_y]])

    return action


def no_operation(obs):
    action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    return action

def move_unit(obs, mode):       # mode= 1,2,3,4 & up,down,left,right

    selected_unit_position_y, selected_unit_position_x = (
                obs[0].observation["screen"][_SELECTED_UNIT] == True).nonzero()
    target_x, target_y = np.mean(selected_unit_position_x), np.mean(selected_unit_position_y)

    if mode == 1:   #up
        dest_x, dest_y = np.clip(target_x, 0, 15), np.clip(target_y - 3, 0, 15)
    elif mode == 2: #down
        dest_x, dest_y = np.clip(target_x, 0, 15), np.clip(target_y + 3, 0, 15)
    elif mode == 3: #left
        dest_x, dest_y = np.clip(target_x - 3, 0, 15), np.clip(target_y, 0, 15)
    elif mode == 4: #right
        dest_x, dest_y = np.clip(target_x + 3, 0, 15), np.clip(target_y, 0, 15)
    action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [dest_x, dest_y]])  # move Up

    return action

def actAgent2Pysc2(i, obs):
    if i == 0:
        action = move_unit(obs, 1)
    elif i == 1:
        action = move_unit(obs, 2)
    elif i == 2:
        action = move_unit(obs, 3)
    elif i == 3:
        action = move_unit(obs, 4)
    elif i ==100:
        action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    return action