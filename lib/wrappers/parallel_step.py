import jax
import jax.numpy as jnp
from typing import Any, Tuple, Optional, Union
import chex
from flax import struct
from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import EnvState, Observation, Level, EnvParams

@struct.dataclass
class ParallelStepLevel:
    level: Level
    par_env_states: EnvState
    par_env_obs   : Observation
    mask_of_levels_to_consider: chex.Array
    
    
@struct.dataclass
class ParallelStepState:
    env_state: EnvState
    
    par_env_states: EnvState
    is_consistent_thus_far: bool
    mask_of_levels_to_consider: bool

def _is_obs_consistent(obs, par_obs, mask_of_levels_to_consider):
    num_envs = jax.tree_util.tree_flatten(par_obs)[0][0].shape[0]
    eq_tree = jax.tree_map(lambda X, y: (X == y).reshape(num_envs, -1).all(axis=-1), par_obs, obs)
    eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
    eq_mask = jnp.array(eq_tree_flat).all(axis=0)
    return (eq_mask & mask_of_levels_to_consider).any()


class ParallelStepWrapper(UnderspecifiedEnv):
    """
    Wraps an environment such that first n steps are taken from some predefined action sequence.
    """
    
    def __init__(self, env: UnderspecifiedEnv):
        self._env = env
        
    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params
    
    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng, _rng = jax.random.split(rng)
        # Normal step
        obs, env_state, reward, done, info = self._env.step(
            _rng,
            state.env_state,
            action,
            params
        )

        # This is the parallel step
        num_envs = jax.tree_util.tree_flatten(state.par_env_states)[0][0].shape[0]
        par_next_obs, par_env_state, par_reward, par_done, par_info = jax.vmap(
            self._env.step, in_axes=(0, 0, None, None)
        )(jax.random.split(rng, num_envs), state.par_env_states, action, params)
        
        mask_to_use = _is_obs_consistent(obs, par_next_obs, state.mask_of_levels_to_consider)

        info['should_train_on_mask'] = ~state.is_consistent_thus_far

        return obs, state.replace(env_state=env_state, par_env_states=par_env_state, is_consistent_thus_far=state.is_consistent_thus_far & mask_to_use), reward, done, info

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: ParallelStepLevel,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level.level, params)
        return obs, ParallelStepState(env_state=env_state, par_env_states=level.par_env_states,
                                      is_consistent_thus_far=_is_obs_consistent(obs, level.par_env_obs, level.mask_of_levels_to_consider),
                                      mask_of_levels_to_consider=level.mask_of_levels_to_consider)
    
    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)