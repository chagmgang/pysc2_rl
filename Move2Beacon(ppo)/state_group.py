import sys

from absl import flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np
import matplotlib.pyplot as plt

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

def obs2state(obs):
    marin_y, marin_x = (obs[0].observation["screen"][_PLAYER_RELATIVE] == friendly).nonzero()
    beacon_y, beacon_x = (obs[0].observation["screen"][_PLAYER_RELATIVE] == neutral).nonzero()
    marin_x, marin_y, beacon_x, beacon_y = np.mean(marin_x), np.mean(marin_y), np.mean(beacon_x), np.mean(beacon_y)

    #return [marin_x*20/63, marin_y*20/63, beacon_x*20/63, beacon_y*20/63]
    return [marin_x*10/63 - beacon_x*10/63,  marin_y*10/63 - beacon_y*10/63]

def obs2distance(obs):
    marin_y, marin_x = (obs[0].observation["screen"][_PLAYER_RELATIVE] == friendly).nonzero()
    beacon_y, beacon_x = (obs[0].observation["screen"][_PLAYER_RELATIVE] == neutral).nonzero()
    marin_x, marin_y, beacon_x, beacon_y = np.mean(marin_x), np.mean(marin_y), np.mean(beacon_x), np.mean(beacon_y)

    now_distance = ((marin_x/63 - beacon_x/63)**2 + (marin_y/63 - beacon_y/63)**2)

    return now_distance