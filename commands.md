1. Training

Basic training (default: 300,000 timesteps)

```bash
python3 train.py
```
Training with custom timesteps

```bash
python3 train.py --timesteps 500000
```
Training with different difficulty levels
```bash
python3 train.py --timesteps 300000
```
# (Difficulty is automatically "medium" in env; change in code if needed)

Resume training from a checkpoint
```bash
python3 train.py --resume runs/ppo_parking/ppo_parking_20000_steps.zip --timesteps 100000
```
Custom log directory

```bash
python3 train.py --logdir runs/my_experiment --timesteps 300000
```
Verify environment before training (sanity check)

```bash
python3 train.py --check_env
```
2. Evaluation
Evaluate trained model (10 episodes, default: medium difficulty)

```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 10
```
Evaluate with fixed start position (no randomization)
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 5 --no_random
```

Evaluate on different difficulty level
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 10 --difficulty hard
```

Test with custom max steps per episode
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 20 --max_steps 1500
```
Evaluate on easy difficulty for sanity check

```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 3 --difficulty easy --no_random
```

3. Typical Workflow

# 1. Start CoppeliaSim and open scene
# (leave it running in the background)

# 2. Check environment is working
```bash
python3 train.py --check_env
```

# 3. Train the model (this may take 10-30 minutes)
```bash
python3 train.py --timesteps 300000 --logdir runs/ppo_parking_v1
```

# 4. Evaluate on easy difficulty to verify
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 5 --difficulty easy
```

# 5. Evaluate on medium difficulty
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 10 --difficulty medium
```

# 6. Evaluate on hard difficulty
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 10 --difficulty hard
```

# 7. Test on fixed position (for debugging)
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 3 --no_random
```


#Tensorboard
```bash
tensorboard --logdir runs/ppo_parking
```