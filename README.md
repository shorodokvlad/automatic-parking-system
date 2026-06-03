<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shorodokvlad/automatic-parking-system">
    <h1 align="center">🚗 CoppeliaSim RL Parallel Parking v1.0.0</h1>
  </a>

<h3 align="center">Automatic Parking System</h3>

  <p align="center">
    A reinforcement learning agent trained to parallel park a car in CoppeliaSim using Stable-Baselines3 (PPO).
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This is a complete Reinforcement Learning-based Automatic Parking System. It features a custom Gymnasium environment representing an ego vehicle equipped with proximity sensors and a vision sensor that must learn to parallel park between two stationary vehicles in CoppeliaSim. The simulator communicates via the ZeroMQ Remote API client.

Key features include:
* **Custom Gymnasium Environment**: Multi-input observation space combining 1D vehicle/proximity state vector and 2D camera images.
* **Stable-Baselines3 PPO**: State-of-the-art proximal policy optimization agent.
* **ZeroMQ Remote API Integration**: High-performance socket-based interface to the CoppeliaSim 3D physics engine.
* **Dynamic Gap & Starting Pose**: Randomized parking spot spacing and starting orientations to ensure robust generalization.
* **Configurable Difficulty Levels**: Easy, Medium, and Hard presets that programmatically modify the required parking gap size.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-shield]][Python-url]
* [![PyTorch][PyTorch-shield]][PyTorch-url]
* [![Stable Baselines 3][SB3-shield]][SB3-url]
* [![Gymnasium][Gymnasium-shield]][Gymnasium-url]
* [![CoppeliaSim][CoppeliaSim-shield]][CoppeliaSim-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to get a copy of the project up and running locally for training and evaluation.

### Prerequisites

You need to install the following on your machine:
* **Python 3.8 - 3.12**
* **CoppeliaSim** (Edu or Player edition)

### Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/shorodokvlad/automatic-parking-system.git
   cd automatic-parking-system
   ```
2. **Setup Virtual Environment**
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   *Alternatively, install the project as an editable package:*
   ```sh
   pip install -e .
   ```
4. **CoppeliaSim Setup**
   * Open **CoppeliaSim**.
   * Open the scene file `Parking.ttt` (located in the project root directory).
   * Ensure that CoppeliaSim is running and listening on the default Remote API port (`23000`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### 1. Training

#### 1.1 Basic Training (Default: 300,000 timesteps, medium difficulty)
```bash
python3 train.py
```

#### 1.2 Training with Custom Timesteps
```bash
python3 train.py --timesteps 500000
```

#### 1.3 Training with Different Difficulty Levels
Configure the environment's parking gap width dynamically by choosing between `easy`, `medium`, or `hard`.
```bash
python3 train.py --difficulty hard --timesteps 300000
```

#### 1.4 Resume Training from a Checkpoint
```bash
python3 train.py --resume runs/ppo_parking/ppo_parking_20000_steps.zip --timesteps 100000
```

#### 1.5 Custom Log Directory
```bash
python3 train.py --logdir runs/my_experiment --timesteps 300000
```

#### 1.6 Verify Environment before Training (Sanity Check)
Runs Gymnasium compatibility and SB3 checks without saving models.
```bash
python3 train.py --check_env
```

---

### 2. Evaluation

#### 2.1 Evaluate Trained Model (Default: 10 episodes, medium difficulty)
```bash
python3 evaluate.py --model runs/ppo_parking/best_model
```

#### 2.2 Evaluate with Fixed Start Position (No Randomization)
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 5 --no_random
```

#### 2.3 Evaluate on a Specific Difficulty Level
Test the model's parking capabilities under tight (`hard`) or wide (`easy`) space constraints.
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 10 --difficulty hard
```

#### 2.4 Test with Custom Max Steps per Episode
```bash
python3 evaluate.py --model runs/ppo_parking/best_model --episodes 20 --max_steps 1500
```

---

### 3. Monitoring (TensorBoard)

Launch TensorBoard to visualize training metrics, reward logs, and episode lengths:
```bash
tensorboard --logdir runs/ppo_parking
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Vladislav Shorodok - [@shorodokvlad](https://twitter.com/shorodokvlad) - vlad.shorodoc@gmail.com

Project Link: [https://github.com/shorodokvlad/automatic-parking-system](https://github.com/shorodokvlad/automatic-parking-system)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[SB3-shield]: https://img.shields.io/badge/Stable--Baselines3-000000?style=for-the-badge&logo=cog&logoColor=white
[SB3-url]: https://stable-baselines3.readthedocs.io/
[Gymnasium-shield]: https://img.shields.io/badge/Gymnasium-008080?style=for-the-badge&logo=gymnasium&logoColor=white
[Gymnasium-url]: https://gymnasium.farama.org/
[CoppeliaSim-shield]: https://img.shields.io/badge/CoppeliaSim-4F8A10?style=for-the-badge&logo=robot&logoColor=white
[CoppeliaSim-url]: https://www.coppeliarobotics.com/
