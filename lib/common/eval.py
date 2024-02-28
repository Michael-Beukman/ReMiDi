import json
import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import orbax.checkpoint as ocp


from lib.common.ppo import ActorCritic
from jaxued.environments.maze.level import Level
from jaxued.environments.maze.util import make_level_generator
from lib.utils import save_compressed_pickle

def evaluate_single(config, eval_env, env_params, rng, train_state):
    sample_0b_level  = make_level_generator(13, 13, 0)
    sample_25b_level = make_level_generator(13, 13, 25)
    
    rng, rng_reset = jax.random.split(rng)
    dr_0b_level = sample_0b_level(rng_reset)
    dr_25b_level = sample_25b_level(rng_reset)
    levels = Level.load_prefabs(config["eval_levels"])

    levels = jax.tree_map(lambda x, y: jnp.append(x, jnp.array([y]), axis=0), levels, dr_0b_level)
    levels = jax.tree_map(lambda x, y: jnp.append(x, jnp.array([y]), axis=0), levels, dr_25b_level)
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]


    if config['lever_game']:
        from lib.environments.lever_game.env import Level as LeverLevel
        num_levels = 25
        levels = LeverLevel(
            correct_answer=jnp.arange(25),
            visible_answer=jnp.ones(25, dtype=jnp.bool_),
            arbitrary_number=jnp.ones(25)
        )
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
    init_hstate = ActorCritic.initialize_carry((num_levels,))
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, cum_reward = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate, init_hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            eval_env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        cum_reward = cum_reward + mask * reward

        return (rng, hstate, obs, next_state, done, next_mask, cum_reward), (state, mask)
    
    (_, _, _, _, _, _, cum_reward), (states, mask) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels),
        ),
        None,
        length=256,
    )
    
    return cum_reward, states, mask


def update_metrics(metrics, levels, config):
    if config["tmaze"]:
        metrics.update({"num_tmaze": levels.is_tmaze.sum() / config["num_train_envs"]})
    if config["blindfold"]:
        metrics.update({"num_blindfold": levels.is_blind.sum() / config["num_train_envs"]})
    if config['lever_game']:
        metrics.update({"num_visible_levers": levels.visible_answer.sum() / config["num_train_envs"]})
    return metrics

def log_eval_main(stats, train_state, config, level_sampler, env_vis, env, env_params, log_sampler_data=True, run_wandb_log=True):
    sampler = train_state.sampler
    
    print(f"Logging update: {stats['update_count']}")
    log_dict = {
        "num_updates": stats["update_count"],
        "num_env_steps": stats["update_count"] * config["num_train_envs"] * config["num_steps"],
    }
    
    # evalution performance
    all_solved = jnp.where(stats["eval_rewards"] > 0, 1., 0.)
    if config['lever_game']:
        eval_level_names = [f'Vis_{i}' for i in range(25)]
    else:
        eval_level_names = config["eval_levels"] + ["dr_0b", "dr_25b"]
    log_dict.update({
        f"solve_rate/{name}": solved.mean() for name, solved in zip(eval_level_names, all_solved)
    })
    log_dict.update({"solve_rate/mean": all_solved.mean()})
    
    # level sampler stats
    if log_sampler_data and sampler is not None:
        log_dict.update({"level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),})
    if sampler is not None:
        log_dict.update({
            "level_sampler/proportion_filled": level_sampler._proportion_filled(sampler),
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
        })

        good = sampler['scores'] > -jnp.inf
        if not config['lever_game']: log_dict.update({"level_sampler/percentage_walls": ((sampler["levels"].wall_map * good[..., None, None]).sum(axis=-1).sum(axis=-1).sum() / good.sum())})
        
        if config['tmaze']:
            log_dict.update({"level_sampler/percentage_tmaze": ((sampler["levels"].is_tmaze * good).sum() / good.sum())})

        
        def get_sampler_stats(sampler):
            good = sampler['scores'] > -jnp.inf
            # get the max score:
            scores = sampler['scores']
            max_score = jnp.where(good, scores, -jnp.inf).max()
            min_score = jnp.where(good, scores, +jnp.inf).min()
            delta = max_score - min_score
            avg_score = (scores * good).sum() / good.sum()
            var = (((scores - avg_score) ** 2) * good).sum() / good.sum()

            duration_inside = sampler['number_of_replays'] - sampler['timestamp_inserted']

            min_duration = jnp.where(good, duration_inside, +jnp.inf).min()
            max_duration = jnp.where(good, duration_inside, -jnp.inf).max()
            avg_duration = (duration_inside * good).sum() / good.sum()

            return {
                "max_score": max_score,
                "min_score": min_score,
                "avg_score": avg_score,
                'max_minus_min': delta,
                "variance": var,
                
                "min_duration":      min_duration,
                "max_duration":      max_duration,
                "avg_duration":      avg_duration,
                "number_of_replays": sampler['number_of_replays'],

                "proportion_filled": level_sampler._proportion_filled(sampler),

            }

        def log_and_plot_single_sampler(sampler, num_images=config['num_images_to_plot']):
            weights_for_levels, w_score, w_staleness = level_sampler.level_weights(sampler, return_subscores=True)
            mask = jnp.arange(level_sampler.capacity) < sampler["size"]
            scores_proper = sampler['scores']
            # we want highest scoring levels first
            top_levels_indices = jnp.argsort(-jnp.where(mask, weights_for_levels, -jnp.inf))[:num_images]
            levels = jax.tree_map(lambda x: x[top_levels_indices], sampler['levels'])
            
            _, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(jax.random.split(jax.random.PRNGKey(0), num_images), levels, env_params)

            overall_scores      = weights_for_levels[top_levels_indices]
            overall_w_score     = w_score[top_levels_indices]
            overall_w_staleness = w_staleness[top_levels_indices]
            buffer_images = []
            for i in range(num_images):
                score        = weights_for_levels[top_levels_indices[i]] # overall_scores[i]
                subscore     = w_score[top_levels_indices[i]] # overall_w_score[i]
                substaleness = w_staleness[top_levels_indices[i]] # overall_w_staleness[i]
                subsub       = scores_proper[top_levels_indices[i]]
                cap = f"{score:.2g} (rnk={subscore:.2g}; stl={substaleness:.2g}; sc={subsub:.2g})"
                buffer_images.append(wandb.Image(env_vis.get_frame(jax.tree_map(lambda x: x[i], init_env_state), env_params), 
                                                 caption=cap))
            return buffer_images
        
        if stats["update_count"] % config['eval_image_freq'] == 0:
            # log the samplers
            # first, do we have one sampler or multiple
            if len(sampler['scores'].shape) == 1:
                # we have one sampler
                log_dict.update({"level_sampler_images/sampler:0": log_and_plot_single_sampler(sampler)})
            else:
                # we have multiple samplers, so log each separately
                for i in range(sampler['scores'].shape[0]):
                    log_dict.update({f"level_sampler_images/sampler:{i}": log_and_plot_single_sampler(jax.tree_map(lambda x: x[i], sampler))})
        
        if len(sampler['scores'].shape) == 1:
            log_dict.update({"level_sampler_stats/sampler:0": get_sampler_stats(sampler)})
        else:
            for i in range(sampler['scores'].shape[0]):
                log_dict.update({f"level_sampler_stats/sampler:{i}": get_sampler_stats(jax.tree_map(lambda x: x[i], sampler))})
    
    
    # training loss
    loss, (critic_loss, actor_loss, entropy) = stats["losses"]
    log_dict.update({
        "agent/loss": loss,
        "agent/critic_loss": critic_loss,
        "agent/actor_loss": actor_loss,
        "agent/entropy": entropy,
    })
    
    dr_inds = (~stats["replay"]).nonzero()[0]
    replay_inds = stats["replay"].nonzero()[0]
    
    # level complexity
    if len(dr_inds) > 0:
        if not config['lever_game']: log_dict.update({"level_complexity/dr_mean_num_blocks": stats["mean_num_blocks"][dr_inds[-1]]})
        if config["tmaze"]: log_dict.update({"level_complexity/dr_num_tmaze": stats["num_tmaze"][dr_inds[-1]]})
        if config["blindfold"]: log_dict.update({"level_complexity/dr_num_blindfold": stats["num_blindfold"][dr_inds[-1]]})
        if config["lever_game"]: log_dict.update({"level_complexity/dr/num_visible_levers": stats["num_visible_levers"][dr_inds[-1]]})
    
    if len(replay_inds) > 0:
        if not config['lever_game']: log_dict.update({"level_complexity/replay_mean_num_blocks": stats["mean_num_blocks"][replay_inds[-1]]})
        if config["tmaze"]: log_dict.update({"level_complexity/replay_num_tmaze": stats["num_tmaze"][replay_inds[-1]]})
        if config["blindfold"]: log_dict.update({"level_complexity/replay_num_blindfold": stats["num_blindfold"][replay_inds[-1]]})
        if config["lever_game"]: log_dict.update({"level_complexity/replay/num_visible_levers": stats["num_visible_levers"][replay_inds[-1]]})

    
    # images and videos
    if stats["update_count"] % config['eval_image_freq'] == 0:
        if len(dr_inds) > 0:
            scores_to_use = jax.tree_map(lambda x: x[dr_inds[-1]], stats["updated_scores"])
            states = jax.tree_map(lambda x: x[dr_inds[-1]], stats["levels_played"])
            states = [jax.tree_map(lambda x: x[i], states) for i in range(config["num_train_envs"])]
            images = []
            for index, state in enumerate(states):
                images.append(wandb.Image(env_vis.get_frame(state, env_params), caption=f"Score: {scores_to_use[index]:.2f}"))
            log_dict.update({"images/dr_levels": images})
        
        if len(replay_inds) > 0:
            scores_to_use = jax.tree_map(lambda x: x[replay_inds[-1]], stats["updated_scores"])
            states = jax.tree_map(lambda x: x[replay_inds[-1]], stats["levels_played"])
            states = [jax.tree_map(lambda x: x[i], states) for i in range(config["num_train_envs"])]
            images = []
            for index, state in enumerate(states):
                images.append(wandb.Image(env_vis.get_frame(state, env_params), caption=f"Score: {scores_to_use[index]:.2f}"))
            log_dict.update({"images/replay_levels": images})
        
        if stats["update_count"] % config['eval_video_freq'] == 0:
            print("Recording rollouts")
            for i, level_name in enumerate(config["eval_levels"] + ["dr_0b", "dr_25b"]):
                states, mask = jax.tree_map(lambda x: x[i], stats["rollout"])
                states = [jax.tree_map(lambda x: x[i], states) for i in range(len(mask))]
                frames = []
                for state, mask in zip(states, mask):
                    if not mask:
                        break
                    frames.append(env_vis.get_frame(state.env_state, env_params))
                frames = np.array(frames).transpose(0, 3, 1, 2)
                video = wandb.Video(frames, fps=4)
                log_dict.update({f"animations/{level_name}": video})
    
    if run_wandb_log:
        wandb.log(log_dict)
    
    return log_dict

def setup_checkpointing(config, train_state, env, env_params) -> ocp.CheckpointManager:
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    # and the initial train state
    save_compressed_pickle(os.path.join(overall_save_dir, 'init_train_state.pkl'), {
        'train_state': flax.serialization.to_state_dict(train_state),
        'level_sampler': train_state.sampler,
        'env': env,
        'network': ActorCritic(env.action_space(env_params).n)
    })
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            max_to_keep=65
        )
    )
    return checkpoint_manager


def setup_wandb(config):
    run = wandb.init(config=config, project=config["project"], group=config["run_name"])
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")

    return config, run