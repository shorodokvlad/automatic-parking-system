# Create virtual enviroment
```python
python -m venv .venv;
# activation (Windows):
 .venv\Scripts\activate;
# activation (Linux/macOS): 
source .venv/bin/activate;
```
## Install the packages listed inside a requirements.txt
```
pip install -r requirements.txt
```
# 1. Training

## 1.1 Basic training (default: 300,000 timesteps)

```bash
python3 train.py
```
## 1.2 Training with custom timesteps

```bash
python3 train.py --timesteps 500000
```
## 1.3 Training with different difficulty levels 
(Difficulty is automatically "medium" in env; change in code if needed)
```bash
python3 train.py --timesteps 300000
```

## 1.4 Resume training from a checkpoint
```bash
python3 train.py --resume runs/ppo_parking/ppo_parking_20000_steps.zip --timesteps 100000
```
## 1.5 Custom log directory

```bash
python3 train.py --logdir runs/my_experiment --timesteps 300000
```
## 1.6 Verify environment before training (sanity check)

```bash
python3 train.py --check_env
```
# 2. Evaluation
## 2.1 Evaluate trained model (10 episodes, default: medium difficulty)

```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 10
```
## 2.2 Evaluate with fixed start position (no randomization)
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 5 --no_random
```

## 2.3 Evaluate on different difficulty level
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 10 --difficulty hard
```

## 2.4 Test with custom max steps per episode
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 20 --max_steps 1500
```
## 2.5 Evaluate on easy difficulty for sanity check

```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 3 --difficulty easy --no_random
```

# 3. Typical Workflow

## 3.1 Start CoppeliaSim and open scene (leave it running in the background)

## 3.2 Check environment is working
```bash
python3 train.py --check_env
```

## 3.3 Train the model (this may take 10-30 minutes)
```bash
python3 train.py --timesteps 300000 --logdir runs/ppo_parking_v1
```

## 3.4 Evaluate on easy difficulty to verify
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 5 --difficulty easy
```

## 3.5 Evaluate on medium difficulty
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 10 --difficulty medium
```

## 3.6 Evaluate on hard difficulty
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 10 --difficulty hard
```

## 3.7 Test on fixed position (for debugging)
```bash
python3 evaluate.py --model runs/ppo_parking_v1/best_model --episodes 3 --no_random
```


# Tensorboard
```bash
tensorboard --logdir runs/ppo_parking
```