def setup_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === PPO ===
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_annealing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--num_updates", type=int, default=30_000)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--num_train_envs", type=int, default=32)
    parser.add_argument("--num_minibatches", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epoch_ppo", type=int, default=5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.98)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    parser.add_argument("--critic_coeff", type=float, default=0.5)
    # === ENV CONFIG ===
    parser.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    parser.add_argument("--n_walls", type=int, default=25)
    parser.add_argument("--randomize_n_walls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--on_new_levels_multiple", action=argparse.BooleanOptionalAction, default=False) # incompatible with any PLR methods.
    parser.add_argument("--n_walls_min", type=int, default=0)
    parser.add_argument("--n_walls_max", type=int, default=100)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_image_freq", type=int, default=1000)
    parser.add_argument("--eval_video_freq", type=int, default=6000)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--num_images_to_plot", type=int, default=32)
    parser.add_argument("--eval_levels", nargs='+', default=["SixteenRooms", "Labyrinth", "StandardMaze"])
    # === PLR ===
    parser.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl", "perfect_regret"])
    parser.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--level_buffer_capacity", type=int, default=4000)
    parser.add_argument("--replay_prob", type=float, default=0.5)
    parser.add_argument("--staleness_coeff", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    parser.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    parser.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    
    # === Env ===
        # === TMaze ===
    parser.add_argument("--tmaze",                      action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tmaze_goal_reward",          type=float, default=1.0)
    parser.add_argument("--tmaze_not_goal_reward",      type=float, default=0.0)
    parser.add_argument("--tmaze_probability",          type=float, default=0.5)
        # === Blindfold ===
    parser.add_argument("--blindfold",                                 action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blindfold_minus_one",                       action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blindfold_ten",                             action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blindfold_probability",                     type=float, default=0.5)
    parser.add_argument("--blindfold_impossible_probability",          type=float, default=0.0)
            # === Lever Game ===
    parser.add_argument("--lever_game", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lever_game_actions", type=int, default=25)
    parser.add_argument("--lever_game_reward_correct", type=float, default=1.0)
    parser.add_argument("--lever_game_reward_incorrect", type=float, default=-1.0)
    parser.add_argument("--lever_game_multiplier_invisible", type=float, default=10.0)
    parser.add_argument("--lever_game_visible_answer_probability", type=float, default=0.5)
    return parser