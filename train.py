"""
train.py
--------
Train a PPO agent to parallel-park.
Usage: python3 train.py
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from parking_env import ParallelParkingEnv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int,   default=300_000)
    p.add_argument("--resume",    type=str,   default=None)
    p.add_argument("--logdir",    type=str,   default="runs/ppo_parking")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # Note: randomise_gap is now True so the AI learns dynamic environments!
    train_env = Monitor(
        ParallelParkingEnv(randomise_start=True, randomise_gap=True),
        filename=os.path.join(args.logdir, "train_monitor"),
    )

    eval_env = Monitor(
        ParallelParkingEnv(randomise_start=True, randomise_gap=False),
        filename=os.path.join(args.logdir, "eval_monitor"),
    )

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=70.0, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = args.logdir,
        log_path             = args.logdir,
        eval_freq            = 5_000,
        n_eval_episodes      = 5,
        callback_on_new_best = stop_cb,
        verbose              = 1,
    )

    checkpoint_cb = CheckpointCallback(save_freq=20_000, save_path=args.logdir, name_prefix="ppo_parking")

    policy_kwargs = dict(net_arch=[128, 128])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env             = train_env,
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            tensorboard_log = args.logdir,
        )
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            tensorboard_log = args.logdir,
        )

    print(f"Training for {args.timesteps} timesteps…")
    model.learn(
        total_timesteps     = args.timesteps,
        callback            = [eval_cb, checkpoint_cb],
        reset_num_timesteps = args.resume is None,
    )

    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"Saved final model → {final_path}.zip")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()