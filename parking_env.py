"""
parking_env.py
--------------
Gymnasium environment: ego car must parallel park between two stationary cars.
Features: 17D Observation Space (with sensors), Dynamic Gap Calculation.
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from coppeliasim_car import AckermannCar, connect

WORLD_SCALE      = 5.0    
MAX_STEPS        = 500    
SIM_DT           = 0.05   

# DEMANDING PERFECTION
POS_THRESHOLD    = 0.10   # Must get completely into the spot
YAW_THRESHOLD    = math.radians(4)
SPEED_THRESHOLD  = 0.3    

COLLISION_DIST   = 0.25   

EGO_NAME   = "/nakedAckermannSteeringCar[0]"
PARK1_NAME = "/nakedAckermannSteeringCar[1]"
PARK2_NAME = "/nakedAckermannSteeringCar[2]"

class ParallelParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomise_start: bool = True, randomise_gap: bool = False, render_mode = None):
        super().__init__()
        self.randomise_start = randomise_start
        self.randomise_gap   = randomise_gap

        self._client, self._sim = connect()
        sim = self._sim

        # Only Ego gets the sensors (fixes the terminal warnings)
        self.ego   = AckermannCar(sim, EGO_NAME, has_sensors=True)
        self.park1 = AckermannCar(sim, PARK1_NAME, has_sensors=False)
        self.park2 = AckermannCar(sim, PARK2_NAME, has_sensors=False)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._step_count  = 0
        self._prev_dist   = None
        self._target      = np.array([0.0, 0.0], dtype=np.float32)
        self._target_yaw  = 0.0

    def _update_target_from_parked_cars(self):
        """Calculates the dynamic target spot based on parked car positions."""
        p1_pos, p1_yaw = self.park1.get_pose() 
        p2_pos, p2_yaw = self.park2.get_pose() 

        # The perfect parking spot is exactly halfway between them
        target_x = (p1_pos[0] + p2_pos[0]) / 2.0
        target_y = (p1_pos[1] + p2_pos[1]) / 2.0

        self._target = np.array([target_x, target_y], dtype=np.float32)
        # Assume the ego car should face the same way as the front parked car
        self._target_yaw = p1_yaw

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sim = self._sim
        sim.stopSimulation()
        time.sleep(0.3)
        sim.startSimulation()
        time.sleep(0.3)

        if self.randomise_gap:
            self._randomise_gap()

        # Update the math target before spawning the ego car
        self._update_target_from_parked_cars()

        if self.randomise_start:
            ego_x, ego_y, ego_yaw = self._sample_start()
        else:
            # Fixed start relative to the dynamic gap
            ego_x = self._target[0]
            ego_y = self._target[1] - 1.5
            ego_yaw = self._target_yaw

        self.ego.set_pose(ego_x, ego_y, ego_yaw)
        self.ego.stop()

        self._step_count = 0
        self._prev_dist  = None

        obs = self._get_obs()
        self._prev_dist = self._dist_to_target(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        steer = float(np.clip(action[0], -1, 1))
        speed = float(np.clip(action[1], -1, 1))

        self.ego.set_controls(steer, speed)
        self._client.step()          
        self._step_count += 1

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(obs)
        truncated = (self._step_count >= MAX_STEPS)

        if terminated or truncated:
            self.ego.stop()

        self._prev_dist = self._dist_to_target(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._sim.stopSimulation()
        except Exception:
            pass

    def _get_obs(self) -> np.ndarray:
        ego_pos, ego_yaw = self.ego.get_pose()
        ego_vel          = self.ego.get_velocity()   
        p1_pos, _        = self.park1.get_pose()
        p2_pos, _        = self.park2.get_pose()

        target = self._target
        t_yaw  = self._target_yaw

        to_target = (target - ego_pos) / WORLD_SCALE
        to_p1     = (p1_pos  - ego_pos) / WORLD_SCALE
        to_p2     = (p2_pos  - ego_pos) / WORLD_SCALE

        yaw_err   = _angle_wrap(t_yaw - ego_yaw) / math.pi
        time_left = 1.0 - self._step_count / MAX_STEPS

        sensors = self.ego.get_proximity_readings(max_distance=3.0)
        s_front = sensors["front"] / 3.0
        s_left  = sensors["left"] / 3.0
        s_right = sensors["right"] / 3.0
        s_back  = sensors["back"] / 3.0

        obs = np.array([
            ego_pos[0] / WORLD_SCALE,   # 0
            ego_pos[1] / WORLD_SCALE,   # 1
            ego_yaw / math.pi,          # 2
            ego_vel[0] / 5.0,           # 3  vx
            ego_vel[1] / 5.0,           # 4  vy
            ego_vel[2] / math.pi,       # 5  omega
            to_target[0],               # 6
            to_target[1],               # 7
            yaw_err,                    # 8
            to_p1[0],                   # 9
            to_p1[1],                   # 10
            to_p2[0],                   # 11
            to_p2[1],                   # 12
            time_left,                  # 13
            s_front,                    # 14 
            s_left,                     # 15 
            s_right,                    # 16 
            s_back,                     # 17 <--- Make sure this is actually here!
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self, obs: np.ndarray):
        ego_pos, ego_yaw = self.ego.get_pose()
        p1_pos, _        = self.park1.get_pose()
        p2_pos, _        = self.park2.get_pose()

        dist_to_target = self._dist_to_target(obs)
        yaw_err        = abs(_angle_wrap(self._target_yaw - ego_yaw))

        col1 = np.linalg.norm(ego_pos - p1_pos)
        col2 = np.linalg.norm(ego_pos - p2_pos)
        colliding = (col1 < COLLISION_DIST) or (col2 < COLLISION_DIST)

        progress = 0.0
        if self._prev_dist is not None:
            progress = (self._prev_dist - dist_to_target) * 15.0   

        r_align = -yaw_err / math.pi * 2.5
        r_col   = -1.0 if colliding else 0.0
        r_time  = -0.001

        reward = progress + r_align + r_col + r_time

        terminated = False
        info = {"dist": dist_to_target, "yaw_err": math.degrees(yaw_err)}

        ego_vel = self.ego.get_velocity()
        speed   = np.linalg.norm(ego_vel[:2])

        success = (
            dist_to_target < POS_THRESHOLD
            and yaw_err    < YAW_THRESHOLD
            and speed      < SPEED_THRESHOLD
        )

        if success:
            reward    += 20.0
            terminated = True
            info["result"] = "success"
        elif colliding:
            reward    -= 5.0
            terminated = True
            info["result"] = "collision"

        return reward, terminated, info

    def _dist_to_target(self, obs: np.ndarray) -> float:
        dx = obs[6] * WORLD_SCALE
        dy = obs[7] * WORLD_SCALE
        return math.hypot(dx, dy)

    def _sample_start(self):
        """Phase 2: Realistic parallel parking start position."""
        rng = self.np_random
        p1_pos, p1_yaw = self.park1.get_pose() 

        # Spawn safely in the driving lane
        x   = rng.uniform(p1_pos[0] - 3.0, p1_pos[0] - 2.5) 
        y   = rng.uniform(p1_pos[1] - 1.0, p1_pos[1] + 0.0) 
        yaw = rng.uniform(p1_yaw - 0.1, p1_yaw + 0.1) 
        
        return x, y, yaw

    def _randomise_gap(self):
        rng = self.np_random
        gap_noise1 = rng.uniform(-0.1, 0.1)
        gap_noise2 = rng.uniform(-0.1, 0.1)
        base_p1, base_yaw1 = self.park1.get_pose()
        base_p2, base_yaw2 = self.park2.get_pose()
        self.park1.set_pose(base_p1[0], base_p1[1] + gap_noise1, base_yaw1)
        self.park2.set_pose(base_p2[0], base_p2[1] + gap_noise2, base_yaw2)

def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi