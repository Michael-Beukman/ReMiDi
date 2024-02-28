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
class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def main(config=None):
    config, run = setup_wandb(config)
    if config['lever_game']:
        from lib.common.ppo import ActorCriticFeedForward as ActorCritic
    else:
        from lib.common.ppo import ActorCritic
        
    def log_eval(stats, train_state):
        log_eval_main(stats, train_state, config, level_sampler, env_vis, env._env, env_params, log_sampler_data=True, run_wandb_log=True)
        
    env, env_vis, eval_env, env_params, sample_random_level = get_environment_and_params(config)

    level_sampler = get_level_sampler(config, duplicate_check=config['buffer_duplicate_check'])


    def create_train_state(rng):
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        init_hstate = ActorCritic.initialize_carry((config["num_train_envs"],))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, init_hstate, init_hstate)

        tx = create_optimizer(config)
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
        )

    def train(rng, train_state):
        def on_replay_levels(rng, train_state):
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
            )
            

            metrics = {
                "replay": True,
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "levels_played": init_env_state.env_state,
                "updated_scores": scores
            }
            if not config['lever_game']:
                metrics.update({"mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],})
            metrics = update_metrics(metrics, levels, config)
            return (rng, train_state.replace(sampler=sampler)), metrics
            
        def on_new_levels(rng, train_state):
            sampler = train_state.sampler
            
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
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
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            sampler, scores = insert_to_level_sampler(config, env._env, env_params, level_sampler, sampler, new_levels, values, advantages, dones, rewards, own_extras_to_add={})

            
            # train_state only modified if exploratory_grad_updates is on
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
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "replay": False,
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "levels_played": init_env_state.env_state,
                "updated_scores": scores
            }
            if not config['lever_game']:
                metrics.update({"mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"]})
            
            metrics = update_metrics(metrics, new_levels, config)
            return (rng, train_state.replace(sampler=sampler)), metrics

        def train_step(carry, _):
            rng, train_state = carry
            rng, rng_replay = jax.random.split(rng)
            return jax.lax.cond(
                level_sampler.sample_replay_decision(train_state.sampler, rng_replay),
                on_replay_levels,
                on_new_levels,
                rng, train_state
            )
            
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
    config = vars(parser.parse_args())
    main(config)
