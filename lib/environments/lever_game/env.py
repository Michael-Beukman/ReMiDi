from typing import Tuple
from enum import IntEnum
import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces
from jaxued.environments import UnderspecifiedEnv

@struct.dataclass
class Level:
    correct_answer:   int
    visible_answer:   bool
    arbitrary_number: int

@struct.dataclass
class EnvState:
    correct_answer:   int
    visible_answer:   bool
    arbitrary_number: int

@struct.dataclass
class Observation:
    obs: chex.Array

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 2
    
class LeverGame(UnderspecifiedEnv):
    def __init__(self, num_actions: int = 10, reward_correct=1, reward_incorrect=-1, multiplier_invisible=10, **kwargs):
        super().__init__()
        self.num_actions = num_actions
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.multiplier_invisible = multiplier_invisible

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        
        is_correct = (action == state.correct_answer)
        reward = jax.lax.select(is_correct, self.reward_correct, self.reward_incorrect)
        reward = reward * jax.lax.select(state.visible_answer, 1.0, self.multiplier_invisible)
        done = True
        return (
            self.get_obs(state),
            state,
            reward.astype(jnp.float32),
            done,
            {},
        )
    
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = self.init_state_from_level(level)
        return self.get_obs(state), state

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    # ===
    
    def init_state_from_level(self, level):
        return EnvState(
            correct_answer=level.correct_answer,
            visible_answer=level.visible_answer,
            arbitrary_number=level.arbitrary_number
        )
        

    def get_obs(self, state: EnvState):
        out = jnp.zeros(self.num_actions + 1)
        return jax.lax.select(state.visible_answer, out.at[state.correct_answer + 1].set(1.0), out.at[0].set(1.0))

def make_lever_level_generator(num_actions, visible_answer_probability=0.5, **kwargs):
    def generate(rng):
        rngs = jax.random.split(rng, 3)
        arb = jax.random.randint(rngs[0], (), 0, 10000)
        correct_answer = jax.random.randint(rngs[1], (), 0, num_actions)
        visible_answer = jax.random.uniform(rngs[2]) < visible_answer_probability
        return Level(correct_answer=correct_answer, visible_answer=visible_answer, arbitrary_number=arb)
    return generate