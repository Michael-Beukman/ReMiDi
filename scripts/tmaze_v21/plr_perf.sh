export PYTHONPATH=`pwd`
SHARED="--no-use_accel --n_walls 25 --eval_freq 500 --project dcd --num_updates 30000 --lr_annealing --tmaze --tmaze_not_goal_reward -1"
for j in {0..9}; do
    for i in 0; do
        CUDA_VISIBLE_DEVICES=$i python x_minigrid_plr.py --seed $((i+1*j)) --run_name tmaze_plr --replay_prob 0.8 --score_function=perfect_regret --entropy_coeff 0.0 --gae_lambda 0.95 --lr 0.001 --temperature 1.0 $SHARED &
    done
    wait;
done
wait;