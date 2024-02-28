from jaxued.level_sampler import LevelSampler
from lib.utils import compute_max_returns, max_mc, monte_carlo_regret, positive_value_loss
import jax.numpy as jnp
import jax

def get_level_sampler(config, duplicate_check=True):
    return LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": 4},
        duplicate_check=duplicate_check,
    )


def _get_scores_and_extras(config, env, env_params, level_extras, levels, 
                         values, advantages, dones, rewards):
    extras_to_add = {}
    if level_extras is None:
        max_returns = compute_max_returns(dones, rewards)
    else:
        max_returns = jnp.maximum(level_extras["max_return"], compute_max_returns(dones, rewards))
    extras_to_add['max_return'] = max_returns
    if config['score_function'] == 'MaxMC':
        scores = max_mc(dones, values, max_returns)
    elif config['score_function'] == 'pvl':
        scores = positive_value_loss(dones, advantages)
    elif config['score_function'] == 'perfect_regret':
        if config['lever_game']:
            return levels.visible_answer * 1 + config['lever_game_multiplier_invisible'] * (1 - levels.visible_answer), extras_to_add
        optimal_values = jax.vmap(compute_steps_to_goal, (0, None, None, None))(levels, env.max_height, env.max_width, config)
        min_steps = jax.vmap(min_steps_to_goal, (0, 0))(optimal_values, levels)
        N = n = min_steps
        optimal_values = (1.0 - 0.9*((N+1)/env_params.max_steps_in_episode)) * config['gamma'] ** n
        
        if config['tmaze']:
            optimals_if_tmaze = jnp.ones_like(levels.is_tmaze, dtype=jnp.float32) * config['gamma'] ** 2
            optimal_values = jnp.where(levels.is_tmaze, optimals_if_tmaze, optimal_values)
        
        optimal_values = jnp.where(n != jnp.inf, optimal_values, 0)
        scores = monte_carlo_regret(dones, optimal_values, rewards, gamma=config['gamma'])
    return scores, extras_to_add

def update_level_sampler(config, env, env_params, level_sampler, sampler, levels, level_inds,
                         values, advantages, dones, rewards):
    level_extras = level_sampler.get_levels_extra(sampler, level_inds)
    scores, extras_to_add = _get_scores_and_extras(config, env, env_params, level_extras, levels, 
                         values, advantages, dones, rewards)
    sampler = level_sampler.update_batch(sampler, level_inds, scores, {**level_extras, **extras_to_add})
    return sampler, scores

def insert_to_level_sampler(config, env, env_params, level_sampler, sampler, levels, values, advantages, dones, rewards, own_extras_to_add=None, bad_level_mask=None):
    if own_extras_to_add is None: own_extras_to_add = {}
    level_extras = None
    scores, extras_to_add = _get_scores_and_extras(config, env, env_params, level_extras, levels, values, advantages, dones, rewards)
    if bad_level_mask is not None:
        scores = jnp.where(bad_level_mask, -jnp.inf, scores)
    sampler, _ = level_sampler.insert_batch(sampler, levels, scores, {**extras_to_add, **own_extras_to_add})
    return sampler, scores


def compute_steps_to_goal(level, max_height, max_width, config=None):
    wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
    
    def compute_next(values):
        fwd_values = jnp.array([
            jnp.roll(values[0], -1, axis=1).astype(float).at[:,-1].set(jnp.inf),
            jnp.roll(values[1], -1, axis=0).astype(float).at[-1,:].set(jnp.inf),
            jnp.roll(values[2], 1, axis=1).astype(float).at[:,0].set(jnp.inf),
            jnp.roll(values[3], 1, axis=0).astype(float).at[0,:].set(jnp.inf),
        ])
        new_values = jnp.empty_like(values)
        for i in range(4):
            new_values = new_values.at[i].set(jnp.min(
                jnp.array([values[i], values[i-1] + 1, values[(i+1)%4] + 1, fwd_values[i] + 1]), axis=0
            ))
        return jnp.maximum(new_values, wall_values)
    
    def cond_fn(carry):
        values, next_values = carry
        return jnp.any(values != next_values)
    
    def body_fn(carry):
        _, values = carry
        return values, compute_next(values)
    
    values = jnp.full((4, max_height, max_width), jnp.inf).at[:, level.goal_pos[1], level.goal_pos[0]].set(0)
    return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]

def min_steps_to_goal(min_steps_array, state):
    return min_steps_array[state.agent_dir, state.agent_pos[1], state.agent_pos[0]]

