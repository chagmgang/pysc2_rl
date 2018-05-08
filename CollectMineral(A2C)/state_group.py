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
    marine_map = (obs[0].observation["screen"][_PLAYER_RELATIVE] == friendly).astype(int)
    mineral_map = (obs[0].observation["screen"][_PLAYER_RELATIVE] == neutral).astype(int)
    state = np.dstack((marine_map, mineral_map))
    return state