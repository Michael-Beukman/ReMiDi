from typing import Tuple
import chex
from chex import PRNGKey
import jax
from jaxued.environments.underspecified_env import EnvParams, EnvState, Level, Observation, UnderspecifiedEnv
from lib.environments.underspecified_env_wrapper import UnderspecifiedEnvWrapper, WrappedEnvState, WrappedLevel
from flax import struct
import jax.numpy as jnp
from jaxued.environments.maze.level import Level as OGMazeLevel
@struct.dataclass
class BlindFoldWrappedState(WrappedEnvState):
    _env_state: EnvState
    is_blind: bool = False

@struct.dataclass
class BlindFoldWrappedLevel(WrappedLevel):
    _level: Level
    is_blind: bool = False

class BlindFoldWrapper(UnderspecifiedEnvWrapper):
    def __init__(self, env: UnderspecifiedEnv, 
                 make_minus_one=False,
                 make_10=False) -> None:
        super().__init__(env)
        self.make_minus_one = make_minus_one
        self.make_10 = make_10

    def _format_obs(self, obs, is_blind):
        A =  jax.tree_map(
            lambda x: jnp.where(is_blind, x * 0 + 10, x), obs
        )
        if self.make_minus_one:
            A = A.replace(image=A.image - 1)
        elif self.make_10:
            A = A.replace(image=A.image + 10)
        return A
    
    def reset_env_to_level(self, rng: PRNGKey, level: Level, params: EnvParams) -> Tuple[Observation, EnvState]:
        obs, state = self._env.reset_env_to_level(rng, level, params)
        obs = self._format_obs(obs, level.is_blind)
        return obs, BlindFoldWrappedState(_env_state=state, is_blind=level.is_blind)
    
    def step_env(self, rng: PRNGKey, state: BlindFoldWrappedState, action: int | float, params) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:

        obs, new_state, reward, done, info = self._env.step_env(rng, state._env_state, action, params)
        
        obs = self._format_obs(obs, state.is_blind)

        return obs, state.replace(_env_state=new_state), reward, done, info

def make_blind_level_sampler(original_sampler, blind_prob: float = 0.5, impossible_prob: float = 0.0):
    def sample_random(rng: chex.PRNGKey):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        _level     = original_sampler(_rng)

        A = blind_prob + impossible_prob
        number = jax.random.uniform(_rng)
        idx = 0 * ((0 <= number) & (number < impossible_prob)) + 1 * ((impossible_prob <= number) & (number < A)) + 2*(A <= number)
        prison = jax.tree_map(lambda x: x[0], OGMazeLevel.load_prefabs(['Prison']))
        
        prison   = BlindFoldWrappedLevel(_level=prison, is_blind=False)
        level    = BlindFoldWrappedLevel(_level=_level, is_blind=False)
        bf_level = BlindFoldWrappedLevel(_level=_level, is_blind=True)

        return jax.lax.switch(
            idx,
            [
                (lambda: prison),
                (lambda: bf_level),
                (lambda: level),
            ]
        )

    return sample_random