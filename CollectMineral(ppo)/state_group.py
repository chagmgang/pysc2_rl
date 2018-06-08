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
    marine_map = (obs[0].observation["screen"][_SELECTED_UNIT] == friendly)
    mineral_map = (obs[0].observation["screen"][_PLAYER_RELATIVE] == neutral)
    state = np.dstack((marine_map, mineral_map)).reshape(32*32*2)
    return state

def obs2done(obs):
    collected_mineral = (obs[0].observation['player'][1])
    return collected_mineral
