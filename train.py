"""
train.py  (v2)
--------------
Changes from v1:
  - check_env also verifies that sim stepping actually advances observations
  - Smaller n_steps (1024) → more frequent updates
  - Higher gamma (0.995) → long-horizon task
  - Lower ent_coef (0.005) → reward signal is clear, less exploration noise
  - Larger net_arch (256, 256) → more capacity for sin/cos obs
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from parking_env import ParallelParkingEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--logdir",    type=str, default="runs/ppo_parking")
    p.add_argument("--check_env", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # ── Optional sanity check ─────────────────────────────────────────────
    if args.check_env:
        print("Running env checker + stepping sanity check…")
        env = ParallelParkingEnv(randomise_start=False)
        check_env(env, warn=True)

        obs, _ = env.reset()
        obs2, *_ = env.step(env.action_space.sample())
        diff = abs(obs2 - obs).max()
        if diff < 1e-6:
            print("WARNING: obs unchanged after step — check client.setStepping(True).")
        else:
            print(f"OK: max obs change after step = {diff:.4f}")
        env.close()
        return

    # ── Environments ──────────────────────────────────────────────────────
    train_env = Monitor(
        ParallelParkingEnv(randomise_start=True, randomise_gap=False),
        filename=os.path.join(args.logdir, "train_monitor"),
    )
    eval_env = Monitor(
        ParallelParkingEnv(randomise_start=True, randomise_gap=False),
        filename=os.path.join(args.logdir, "eval_monitor"),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=18.0, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=5_000,
        n_eval_episodes=5,
        callback_on_new_best=stop_cb,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=args.logdir,
        name_prefix="ppo_parking",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=[256, 256])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            verbose=1,
            tensorboard_log=args.logdir,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,      # lower: reward signal is denser now
            n_steps=1024,            # shorter rollouts → more frequent updates
            batch_size=64,
            n_epochs=10,
            gamma=0.995,             # higher: parking is long-horizon
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,          # less noise: reward structure is clear
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.logdir,
        )

    print(f"Training for {args.timesteps} timesteps…")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, checkpoint_cb],
        reset_num_timesteps=args.resume is None,
    )

    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"Saved → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()