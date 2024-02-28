from typing import Tuple
from jaxued.environments.maze.level import Level as MazeLevel
from jaxued.environments.maze.env import OBJECT_TO_INDEX, EnvParams, EnvState as MazeEnvState, EnvState, Maze, make_maze_map
from lib.environments.maze.util import pad_maze_map
from jaxued.environments.maze.util import make_level_generator as og_level_generator
import numpy as np
import chex
from flax import struct
import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict

@struct.dataclass
class Level(MazeLevel):
    is_tmaze: bool
    
    @property
    def is_goal_hidden(self):
        return self.is_tmaze

@struct.dataclass
class EnvState(MazeEnvState):
    is_tmaze: bool
    @property
    def is_goal_hidden(self):
        return self.is_tmaze



l = """
#############
#############
#############
#############
#############
#############
#############
#####B^G#####
#############
#############
#############
#############
#############
"""

TMAZE_LEVEL_RIGHT  = MazeLevel.from_str(l.replace("B", '.'))
TMAZE_LEVEL_LEFT   = MazeLevel.from_str(l.replace("G", '.').replace("B", 'G'))

TMAZE_LEVEL_RIGHT_ACTUAL_TMAZE = Level(**to_state_dict(TMAZE_LEVEL_RIGHT), is_tmaze=True)
TMAZE_LEVEL_LEFT_ACTUAL_TMAZE = Level(**to_state_dict(TMAZE_LEVEL_LEFT), is_tmaze=True)

class TMazeGame(Maze):
    """
    This environment is either a normal maze game, or a T-Maze. If it is a T-Maze, then it has three states, with the goal being invisible.
    The episode terminates after one step.
    """
    def __init__(self, reward_goal: float = 1.0, reward_not_goal=0.0, **kwargs):
        super().__init__(**kwargs)
        self.reward_goal = reward_goal
        self.reward_not_goal = reward_not_goal

    def init_state_from_level(self, level):
        maze_map = make_maze_map(level, padding=0)

        goal_x, goal_y = level.goal_pos
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
        
        # Maybe hide the goal:
        maze_map = jax.lax.select(level.is_goal_hidden, maze_map.at[goal_y, goal_x,:].set(empty), maze_map)
        maze_map = pad_maze_map(maze_map, self.agent_view_size-1)

        # Return the Env State
        return EnvState(
            agent_pos=jnp.array(level.agent_pos, dtype=jnp.uint32),
            agent_dir=jnp.array(level.agent_dir, dtype=jnp.uint8),
            goal_pos=jnp.array(level.goal_pos, dtype=jnp.uint32),
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
            is_tmaze=level.is_tmaze
        )
    
    def _step_agent(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams) -> Tuple[EnvState, float]:
        old_agent_pos = state.agent_pos
        new_state, reward = super()._step_agent(key, state, action, params)
        new_agent_pos = new_state.agent_pos

        has_moved = jnp.any(new_agent_pos != old_agent_pos)
        # Terminate after one step if we are in a T-Maze
        new_state = jax.tree_map(lambda new, old:
                                     jax.lax.select(state.is_tmaze & has_moved, new, old), # if it is a T-maze and it moved, we terminate.
                                     new_state.replace(terminal=True), new_state)

        # The reward depends on the t-maze.
        reward_for_tmaze = jax.lax.select(reward > 0, self.reward_goal, jax.lax.select(has_moved, self.reward_not_goal, reward))
        reward = jax.lax.select(state.is_tmaze, reward_for_tmaze, reward)

        return new_state, reward
    

def make_tmaze_level_generator(height: int, width: int, n_walls: int, tmaze_probability: float = 0.5):
    original_sample = og_level_generator(height, width, n_walls)
    def sample(rng: chex.PRNGKey) -> Level:
        rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
        old_level: MazeLevel = original_sample(_rng1)
        
        # Now sample if the level is actually a t-maze.
        is_tmaze = jax.random.uniform(_rng2) < tmaze_probability
        tmaze_left_or_right = jax.random.uniform(rng) < 0.5

        mask = jnp.zeros_like(old_level.wall_map, dtype=jnp.bool_)
        
        tt = l.strip().replace("\n", '').index("^")
        y = tt // 13
        x = tt % 13

        mask = mask.at[y-5:y+5, x-5:x+5].set(True)
        random = (jax.random.uniform(_rng3, shape=mask.shape) < 0.5)
        
        updated_map = jnp.where(mask, TMAZE_LEVEL_LEFT.wall_map, random)

        t_left  = TMAZE_LEVEL_LEFT .replace(wall_map=updated_map)
        t_right = TMAZE_LEVEL_RIGHT.replace(wall_map=updated_map)

        old_level = jax.tree_map(lambda old, left, right: jax.lax.select(
            is_tmaze, (jax.lax.select(tmaze_left_or_right, left, right)), old), 
            old_level, t_left, t_right
        )

        return Level(**to_state_dict(old_level), is_tmaze=is_tmaze)
    
    return sample
