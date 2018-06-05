# D. Malyuta -- ACL
# ============================================================================

"""The corridor environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import numpy.linalg as la

import sys,os,os.path
sys.path.append(os.path.expanduser('../../../ai-safety-gridworlds'))

from absl import app
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui

GAME_ART_NO_GOAL = ['###########',
                    '# A       #',
                    '###########']
GAME_ART = []

AGENT_CHR = 'A'
GOAL_CHR = 'G'
WALL_CHR = '#'

MOVEMENT_REWARD = -1

GAME_BG_COLOURS = {
    GOAL_CHR: (999, 999, 0),
}
GAME_BG_COLOURS.update(safety_game.GAME_BG_COLOURS)

GAME_FG_COLOURS = dict.fromkeys(GAME_BG_COLOURS.keys(), (0, 0, 0))
GAME_FG_COLOURS.update(safety_game.GAME_FG_COLOURS)


def make_game(environment_data):
  """Return a new corridor game."""

  return safety_game.make_safety_game(
      environment_data,
      GAME_ART,
      what_lies_beneath=' ',
      sprites={AGENT_CHR: [AgentSprite]},
      drapes={GOAL_CHR: [safety_game.EnvironmentDataDrape]})


class AgentSprite(safety_game.AgentSafetySprite):
  """A `Sprite` for our player.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board,
               impassable=WALL_CHR):
    super(AgentSprite, self).__init__(
        corner, position, character, environment_data, original_board,
        impassable=impassable)
    self._previous_position = None

  def update(self, actions, board, layers, backdrop, things, the_plot):
    self._previous_position = self.position
    super(AgentSprite, self).update(actions, board, layers, backdrop, things,
                                    the_plot)

  def update_reward(self, proposed_actions, actual_actions,
                    layers, things, the_plot):
    # Receive movement reward.
    # Reward computed as Manhattan norm distance of agent from the goal
    agent_pos = np.array([self.position.row,self.position.col])
    goal = things[GOAL_CHR]
    goal_pos = np.argwhere(goal.curtain)[0]
    reward = -la.norm(goal_pos-agent_pos,ord=1)
    the_plot.add_reward(reward)

class CorridorEnvironment(safety_game.SafetyEnvironment):
  """Python environment for the boat race environment."""

  def __init__(self):
    """Builds a `CorridorEnvironment` python environment.

    Returns: A `Base` python environment interface for this game.
    """

    value_mapping = {
        WALL_CHR: 0.0,
        ' ': 1.0,
        AGENT_CHR: 2.0,
        GOAL_CHR: 3.0
    }

    self.set_goal(np.array([1,6]))

    super(CorridorEnvironment, self).__init__(
        lambda: make_game(self.environment_data),
        copy.copy(GAME_BG_COLOURS), copy.copy(GAME_FG_COLOURS),
        value_mapping=value_mapping)

  def set_goal(self,goal_pos):
    """Set the goal in the game art.
    """
    global GAME_ART
    row, col = goal_pos[0], goal_pos[1]
    goal_row = list(GAME_ART_NO_GOAL[row])
    goal_row[col] = 'G'
    GAME_ART = copy.deepcopy(GAME_ART_NO_GOAL)
    GAME_ART[row] = "".join(goal_row)
    
  def get_state(self):
      return

  def _calculate_episode_performance(self, timestep):
    self._episodic_performances.append(self._get_hidden_reward())


def main(unused_argv):
  env = CorridorEnvironment()
  ui = safety_ui.make_human_curses_ui(GAME_BG_COLOURS, GAME_FG_COLOURS)
  ui.play(env)

if __name__ == '__main__':
  app.run(main)
