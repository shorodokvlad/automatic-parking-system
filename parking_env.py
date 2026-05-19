"""
parking_env.py
--------------
Gymnasium environment: ego car must parallel park between two stationary cars.
State Space: 17 continuous values (now including 3 proximity sensors).
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

# Relaxed Parking-success thresholds
POS_THRESHOLD    = 0.30   # Metres from target centre
YAW_THRESHOLD    = math.radians(15)
SPEED_THRESHOLD  = 0.3    # rad/s wheel speed

COLLISION_DIST   = 0.25   

# Assuming vertical orientation based on the debug log coordinates
TARGET_X   =  1.400   
TARGET_Y   = -3.000
TARGET_YAW =  math.radians(90)   

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

        self.ego   = AckermannCar(sim, EGO_NAME)
        self.park1 = AckermannCar(sim, PARK1_NAME)
        self.park2 = AckermannCar(sim, PARK2_NAME)

        # 17 Dimensions (14 math + 3 sensors)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._step_count  = 0
        self._prev_dist   = None
        self._target      = np.array([TARGET_X, TARGET_Y], dtype=np.float32)
        self._target_yaw  = TARGET_YAW

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sim = self._sim
        sim.stopSimulation()
        time.sleep(0.3)
        sim.startSimulation()
        time.sleep(0.3)

        if self.randomise_gap:
            self._randomise_gap()

        if self.randomise_start:
            ego_x, ego_y, ego_yaw = self._sample_start()
        else:
            ego_x, ego_y, ego_yaw = TARGET_X, TARGET_Y - 1.5, TARGET_YAW

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

        # Read the newly configured sensors
        sensors = self.ego.get_proximity_readings(max_distance=3.0)
        s_front = sensors["front"] / 3.0
        s_left  = sensors["left"] / 3.0
        s_right = sensors["right"] / 3.0

        obs = np.array([
            ego_pos[0] / WORLD_SCALE,   
            ego_pos[1] / WORLD_SCALE,   
            ego_yaw / math.pi,          
            ego_vel[0] / 5.0,           
            ego_vel[1] / 5.0,           
            ego_vel[2] / math.pi,       
            to_target[0],               
            to_target[1],               
            yaw_err,                    
            to_p1[0],                   
            to_p1[1],                   
            to_p2[0],                   
            to_p2[1],                   
            time_left,                  
            s_front,                    
            s_left,                     
            s_right,                    
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
            # Multiplier increased to 15.0 to pull the car harder toward the spot
            progress = (self._prev_dist - dist_to_target) * 15.0   

        r_align = -yaw_err / math.pi * 0.5
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
        """Curriculum Start: Narrow box behind the spot."""
        rng = self.np_random
        x   = rng.uniform(TARGET_X - 0.2, TARGET_X + 0.2) 
        y   = rng.uniform(TARGET_Y - 1.5, TARGET_Y - 1.0)
        yaw = rng.uniform(TARGET_YAW - 0.2, TARGET_YAW + 0.2) 
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