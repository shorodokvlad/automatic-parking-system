"""
evaluate.py
-----------
Usage: python3 evaluate.py --model runs/ppo_parking/best_model
"""

import argparse
from stable_baselines3 import PPO
from parking_env import ParallelParkingEnv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--no_random", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    env   = ParallelParkingEnv(randomise_start=not args.no_random, randomise_gap=True)
    model = PPO.load(args.model, env=env)

    successes = 0
    collisions = 0

    for ep in range(args.episodes):
        obs, _    = env.reset()
        done      = False
        ep_reward = 0.0
        steps     = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1
            done       = terminated or truncated

        result = info.get("result", "timeout")
        if result == "success": successes += 1
        elif result == "collision": collisions += 1

        dist = info.get("dist", float("nan"))
        yaw  = info.get("yaw_err", float("nan"))
        print(f"Ep {ep+1:3d} | {result:10s} | reward={ep_reward:7.2f} | steps={steps:4d} | dist={dist:.3f}m | yaw={yaw:.1f}°")

    total = args.episodes
    print(f"\nResults over {total} episodes:")
    print(f"  Success   : {successes}/{total}")
    print(f"  Collision : {collisions}/{total}")
    print(f"  Timeout   : {total-successes-collisions}/{total}")
    env.close()

if __name__ == "__main__":
    main()