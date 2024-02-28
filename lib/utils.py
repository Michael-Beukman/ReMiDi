import pickle
import bz2
from typing import Any
import jax
import jax.numpy as jnp
import wandb
from jaxued.environments.maze.level import Level
from lib.environments.maze.vis import MazeVisualizer
from jaxued.utils import accumulate_rollout_stats

def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def plot_initial_levels(levels: Level, env, env_params):
    env_vis = MazeVisualizer(env, tile_size=8)
    rng = jax.random.PRNGKey(0)
    obs, state = jax.vmap(env.reset_to_level, (None, 0, None))(rng, levels, env_params)
    num_levels = levels.wall_map.shape[0]
    images = []
    for i in range(num_levels):
        state_to_use = jax.tree_map(lambda x: x[i], state)
        frame = env_vis.get_frame(state_to_use, env_params)
        image = wandb.Image(frame)
        images.append(image)
    wandb.log({"initial_buffer_levels/levels": images})


def compute_max_returns(dones, rewards):
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
    return max_returns

def max_mc(dones, values, max_returns, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, max_returns[None, :] - values, time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)

def positive_value_loss(dones, advantages, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, jnp.maximum(advantages, 0), time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)

def monte_carlo_regret(dones, optimal_values, rewards, gamma=1.0, incomplete_value=-jnp.inf):
    _, step_count = jax.lax.scan(lambda step_count, done: ((step_count + 1) * ~done, step_count), jnp.zeros_like(dones[0], dtype=jnp.uint32), dones)
    discount = gamma ** step_count
    discount_rewards = rewards * discount
    
    mean_discount_returns, _, episode_count = accumulate_rollout_stats(dones, discount_rewards, time_average=False)
    mean_scores = optimal_values - mean_discount_returns
    
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)
