from lib.environments.lever_game.env import LeverGame, make_lever_level_generator
from jaxued.environments.maze.env import Maze
from jaxued.environments.maze.util import make_level_generator
from lib.environments.maze.variants.blindfold_wrapper import BlindFoldWrapper, make_blind_level_sampler
from lib.environments.maze.variants.tmaze import TMazeGame, make_tmaze_level_generator
from lib.environments.maze.vis import MazeVisualizer

from jaxued.wrappers.autoreplay import AutoReplayWrapper


def get_environment_and_params(config, return_general_kwargs=False):
    MAX_WIDTH = MAX_HEIGHT = 13
    env_class = Maze
    env_kws = {}
    get_gen_func = make_level_generator
    get_gen_kws  = {}
    general_kwargs = dict(height=MAX_HEIGHT, width=MAX_WIDTH, n_walls=config["n_walls"])


    if config['tmaze']:
        env_class = TMazeGame
        env_kws = {'reward_goal': config['tmaze_goal_reward'], 'reward_not_goal': config['tmaze_not_goal_reward']}
        get_gen_func = make_tmaze_level_generator
        get_gen_kws = get_gen_kws | {'tmaze_probability': config['tmaze_probability']}

    if config['lever_game']:
        env_class = LeverGame
        env_kws = {
            'num_actions': config['lever_game_actions'],
            'reward_correct': config['lever_game_reward_correct'],
            'reward_incorrect': config['lever_game_reward_incorrect'],
            'multiplier_invisible': config['lever_game_multiplier_invisible'],
        }
        get_gen_func = make_lever_level_generator
        get_gen_kws = {'num_actions': config['lever_game_actions'], 'visible_answer_probability': config['lever_game_visible_answer_probability']}


    wrappers = []
    if config["blindfold"]:
        wrappers.append(
            (BlindFoldWrapper, {'make_minus_one': config['blindfold_minus_one'], 'make_10': config['blindfold_ten']}, make_blind_level_sampler, {'blind_prob': config['blindfold_probability'], 'impossible_prob': config['blindfold_impossible_probability']})
        )

    env = env_class(max_height=MAX_HEIGHT, max_width=MAX_WIDTH, agent_view_size=config["agent_view_size"], normalize_obs=True, **env_kws)
    sample_random_level = get_gen_func(**(general_kwargs | get_gen_kws))

    if len(wrappers) > 0:
        # add wrappers
        for wrapper_env_class, wrapper_env_kwargs, wrapper_level_sampler, wrapper_gen_kwargs in wrappers:
            env = wrapper_env_class(env, **wrapper_env_kwargs)
            sample_random_level = wrapper_level_sampler(sample_random_level, **wrapper_gen_kwargs)

    env_vis = MazeVisualizer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params

    eval_env = AutoReplayWrapper(Maze(max_height=MAX_HEIGHT, max_width=MAX_WIDTH, agent_view_size=config["agent_view_size"], normalize_obs=True))
    
    if config['lever_game']:
        eval_env = env
    
    if return_general_kwargs:
        return env, env_vis, eval_env, env_params, sample_random_level, general_kwargs
    return env, env_vis, eval_env, env_params, sample_random_level
