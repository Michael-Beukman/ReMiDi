import argparse
from typing import Any
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState

import wandb
from lib.common.arguments import setup_arguments
from lib.common.env import get_environment_and_params
from lib.common.plr import get_level_sampler, insert_to_level_sampler, update_level_sampler
from lib.common.ppo import compute_gae, create_optimizer, sample_trajectories_rnn, update_actor_critic_rnn
from lib.common.eval import evaluate_single, log_eval_main, setup_checkpointing, setup_wandb, update_metrics
from jaxued.wrappers.autoreplay import AutoReplayWrapper
from lib.wrappers.parallel_step import ParallelStepLevel, ParallelStepWrapper

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    current_buffer_index: int = 0
    current_number_of_updates_for_this_buffer: int = 0
    has_reached_mmr: bool = False

def main(config=None):
    config, run = setup_wandb(config)
    if config['lever_game']:
        from lib.common.ppo import ActorCriticFeedForward as ActorCritic
    else:
        from lib.common.ppo import ActorCritic
        
    def log_eval(stats, train_state):
        log_eval_main(stats, train_state, config, level_sampler, env_vis, env._env, env_params, log_sampler_data=False, run_wandb_log=True)
        
    env, env_vis, eval_env, env_params, sample_random_level = get_environment_and_params(config)

    level_sampler = get_level_sampler(config, duplicate_check=config['buffer_duplicate_check'])
    

    def create_train_state(rng):

        pholder_level = sample_random_level(rng)
        obs, _ = env.reset_to_level(rng, pholder_level, env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        init_hstate = ActorCritic.initialize_carry((config["num_train_envs"],))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, init_hstate, init_hstate)
        tx = create_optimizer(config)
        
        rng, rng_level = jax.random.split(rng, 2)
        levels = jax.vmap(sample_random_level)(jax.random.split(rng_level, config["num_outer_adversaries"]))
        init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(jax.random.split(rng, config["num_outer_adversaries"]), levels, env_params)
        # Initialize each level sampler
        sampler = jax.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_outer_adversaries"], axis=0),
            level_sampler.initialize(pholder_level, {"max_return": -jnp.inf,
                                                     'init_obs': jax.tree_map(lambda x: x[0], init_obs),
                                                     }),
        )
        
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            current_buffer_index=0,
            current_number_of_updates_for_this_buffer=0,
            has_reached_mmr=False
        )

    def train(rng, train_state):
        def on_replay_levels(rng, train_state, all_init_obs, all_init_states, mask_which_are_good,):
            sampler = train_state.sampler
            
            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            
            wrapped_env = AutoReplayWrapper(ParallelStepWrapper(env._env))
            
            def make_single_level(level):
                return ParallelStepLevel(level=level, par_env_states=all_init_states, par_env_obs=all_init_obs, mask_of_levels_to_consider=mask_which_are_good)
            wrapped_levels = jax.vmap(make_single_level)(levels)

            # this does the rollouts on ALL of the previous levels in the buffer
            init_obs, init_env_state = jax.vmap(wrapped_env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), wrapped_levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                wrapped_env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            
            action_update_mask = info['should_train_on_mask']
            # jax.debug.breakpoint()

            # Here I basically want to say: For each AOH trajectory in this buffer, I should find which parts of the trajectory are consistent with the minimax regret levels. One way to do that is to have a probability distribution

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            
            sampler, scores = update_level_sampler(config, env._env, env_params, level_sampler, sampler, levels, level_inds, values, advantages, dones, rewards)
            
            # Update the policy using trajectories collected from replay levels
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
                agent_action_mask=action_update_mask
            )
            
            levels = levels
            metrics = {
                "replay": True,
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "levels_played": init_env_state.env_state.env_state,
                "updated_scores": scores
            }
            if not config['lever_game']:
                metrics.update({"mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],})
            metrics = update_metrics(metrics, levels, config)
                
            return (rng, train_state.replace(sampler=sampler)), metrics
        
        def on_new_levels(rng, train_state, all_init_obs, all_init_states, mask_which_are_good, exploratory_grad_updates):
            sampler = train_state.sampler
            
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
                        
            wrapped_env = AutoReplayWrapper(ParallelStepWrapper(env._env))
            
            def make_single_level(level):
                return ParallelStepLevel(level=level, par_env_states=all_init_states, par_env_obs=all_init_obs, mask_of_levels_to_consider=mask_which_are_good)
            
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, n_sampl:= config["num_train_envs"]))
            wrapped_levels = jax.vmap(make_single_level)(new_levels)
            

            init_obs, init_env_state = jax.vmap(wrapped_env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), wrapped_levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                wrapped_env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)


            is_bad_levels = ~info['should_train_on_mask'].any(axis=0)

            sampler, scores = insert_to_level_sampler(config, env._env, env_params, level_sampler, sampler, new_levels, values, advantages, dones, rewards, own_extras_to_add={
                'init_obs': init_obs
            }, bad_level_mask=is_bad_levels)

            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=exploratory_grad_updates,
            )
            
            metrics = {
                "replay": False,
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "levels_played": init_env_state.env_state.env_state,
                "updated_scores": scores
            }
            if not config['lever_game']:
                metrics.update({"mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"]})
            metrics = update_metrics(metrics, new_levels, config)
            
            return (rng, train_state.replace(sampler=sampler)), metrics
        
        def on_new_levels_with_exploratory_grad_updates(*args, **kwargs):
            return on_new_levels(*args, **kwargs, exploratory_grad_updates=True)
        
        def on_new_levels_without_exploratory_grad_updates(*args, **kwargs):
            return on_new_levels(*args, **kwargs, exploratory_grad_updates=False)

        def train_step(carry, _):
            rng, train_state = carry
            # Select, at random, which buffer we decide to train on next
            rng, _rng = jax.random.split(rng)
            adv_idx = train_state.current_buffer_index
            train_state = train_state.replace(current_number_of_updates_for_this_buffer=train_state.current_number_of_updates_for_this_buffer + 1)
            
            # Convert the "outer" train state into your standard PLR train state
            inner_sampler = jax.tree_map(lambda x: x[adv_idx], (train_state.sampler))
            inner_train_state = train_state.replace(sampler=inner_sampler)
            
            mask_which_are_good = jnp.arange(level_sampler.capacity)[None, :] < train_state.sampler["size"][:, None]
            good_adv_idxs = jnp.arange(config['num_outer_adversaries']) < adv_idx
            mask_which_are_good = mask_which_are_good * good_adv_idxs[:, None]
            mask_which_are_good = mask_which_are_good.flatten()
            
            rng, _rng = jax.random.split(rng)

            

            new_all_init_obs, all_init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(_rng, config['num_outer_adversaries'] * config['level_buffer_capacity']), 
                jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), train_state.sampler['levels']), env_params)
            all_init_env_state = all_init_env_state.env_state

            rng, rng_replay = jax.random.split(rng)


            def on_new_levels_func(*args, **kwargs):
                if config['exploratory_grad_updates']:
                    return on_new_levels_with_exploratory_grad_updates(*args, **kwargs)
                else:
                    return on_new_levels_without_exploratory_grad_updates(*args, **kwargs)

            should_replay = level_sampler.sample_replay_decision(inner_train_state.sampler, rng_replay)

            (rng, inner_train_state), metrics = jax.lax.cond(
                should_replay, # only do replay on the main buffer
                on_replay_levels,
                on_new_levels_func,
                rng, inner_train_state, new_all_init_obs, all_init_env_state, mask_which_are_good
            )
            
            # Update the stats
            my_inner_sampler = level_sampler.update_extra_scores(inner_train_state.sampler)
            sampler = jax.tree_map(lambda x, y: x.at[adv_idx].set(y), train_state.sampler, my_inner_sampler)
            train_state = inner_train_state.replace(sampler=sampler)

            def get_new_index(current_index, size):
                # Skip over the first one
                new = (current_index + 1) % size
                # new_val = jax.lax.select(new == 0, 1, new)
                is_mmr  = jax.lax.select(jnp.logical_and(new == 1, current_index == 0), True, False)
                
                if config['stay_at_last']:
                    # This one forces to train only on the last buffer once it has reached that.
                    new = jax.lax.select(new == 0, size - 1, new)
                return new, 0, is_mmr
            
            AA, BB, CC = jax.tree_map(lambda old, new: jax.lax.select(
                train_state.current_number_of_updates_for_this_buffer > config['number_of_replays_per_buffer'],
                new, old
            ), (train_state.current_buffer_index, train_state.current_number_of_updates_for_this_buffer, train_state.has_reached_mmr), 
               (get_new_index(train_state.current_buffer_index, config["num_outer_adversaries"]))
            )

            train_state = train_state.replace(
                current_buffer_index=AA,
                current_number_of_updates_for_this_buffer=BB,
                has_reached_mmr=jnp.logical_or(train_state.has_reached_mmr, CC)
            )

            return (rng, train_state), metrics
            
        def train_and_eval_step(runner_state, _):
            rng, update_count, train_state = runner_state
            (rng, train_state), metrics = jax.lax.scan(train_step, (rng, train_state), None, config["eval_freq"])
            update_count += config["eval_freq"]
            
            rng, rng_eval = jax.random.split(rng)
            eval_rewards, states, masks = jax.vmap(evaluate_single, (None, None, None, 0, None))(config, eval_env, env_params, jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)
            states, masks = jax.tree_map(lambda x: x[0].swapaxes(0, 1), (states, masks)) # just render 1 attempt for each level
            
            stats = {
                **metrics,
                "update_count": update_count + config["eval_freq"],
                "losses": jax.tree_map(lambda x: x[-1], metrics["losses"]),
                "eval_rewards": eval_rewards.swapaxes(0, 1), # (num_eval_levels, eval_num_attempts)
                "rollout": (states, masks),
            }
            
            jax.debug.callback(log_eval, stats, train_state)

            runner_state = (rng, update_count, train_state)
            return runner_state, stats["eval_rewards"]
        

        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
        runner_state = (rng, 0, train_state)
        for eval_step in range(config["num_updates"] // config["eval_freq"]):
            runner_state, eval_reward = train_and_eval_step(runner_state, None)
            checkpoint_manager.save(eval_step, runner_state[2])
        
        
        return train_state, eval_reward

    def rollout(rng):
        rng_init, rng_train = jax.random.split(rng)
        train_states = create_train_state(rng_init)
        return train(rng_train, train_states)

    rng = jax.random.PRNGKey(config["seed"])
    with jax.disable_jit(False):
        return rollout(rng)

if __name__=="__main__":
    parser = setup_arguments()
    wandb.login()
    
    # === ReMiDi ===
    parser.add_argument("--num_outer_adversaries",                    type=int, default=16)
    parser.add_argument("--number_of_replays_per_buffer",             type=int, default=1000)
    parser.add_argument("--stay_at_last",                             action=argparse.BooleanOptionalAction, default=True)
    
    config = vars(parser.parse_args())
    main(config)