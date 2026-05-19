"""
parking_env.py  (v2 — reward & obs fixes)
------------------------------------------
Changes from v1:
  1. client.step() instead of sim.step()  — fixes the "teleport" / free-running sim
  2. sin/cos encoding for yaw              — no angle-wrapping ambiguity
  3. Exponential proximity reward          — smooth gradient all the way in
  4. Decoupled success stages              — first learn to reach, then align
  5. Removed speed from success condition  — one less thing to satisfy
  6. Tighter normalisation                 — obs always in [-1, 1]
  7. Collision uses proper world-frame dist (not obs-space)
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_car import AckermannCar, connect

# ── Scene constants — CALIBRATE THESE TO YOUR SCENE ──────────────────────────

# Centre of the parking gap in world frame
TARGET_X   =  0.150
TARGET_Y   =  0.390
TARGET_YAW =  math.radians(90)   # desired heading (match the parked cars)

EGO_NAME   = "/nakedAckermannSteeringCar[0]"
PARK1_NAME = "/nakedAckermannSteeringCar[1]"
PARK2_NAME = "/nakedAckermannSteeringCar[2]"

# ── Training constants ────────────────────────────────────────────────────────

MAX_STEPS      = 500
WORLD_SCALE    = 3.0    # normalisation radius (metres); keep > max expected dist

# Thresholds for success
POS_THRESHOLD  = 0.30   # metres
YAW_THRESHOLD  = math.radians(15)

# Collision: distance between ego centre and parked-car centre
COLLISION_DIST = 0.30   # metres — tune to your car's body size


class ParallelParkingEnv(gym.Env):
    """
    Gymnasium env for parallel parking with nakedAckermannSteeringCar.

    Observation (14 floats):
      [0]    dist to target / WORLD_SCALE              (0 -> 1)
      [1-2]  unit vector ego->target  (cos th, sin th) (-1 -> 1)
      [3]    cos(yaw_error)                            (-1 -> 1)
      [4]    sin(yaw_error)                            (-1 -> 1)
      [5-6]  ego velocity (vx, vy) / MAX_WHEEL_SPEED  (-1 -> 1)
      [7]    ego angular velocity / pi                 (-1 -> 1)
      [8-9]  unit vector ego->park1  (cos, sin)        (-1 -> 1)
      [10]   dist ego->park1 / WORLD_SCALE             (0 -> 1)
      [11-12]unit vector ego->park2  (cos, sin)        (-1 -> 1)
      [13]   dist ego->park2 / WORLD_SCALE             (0 -> 1)

    Action (2 floats in [-1, 1]):
      [0]  steer_norm
      [1]  speed_norm  (dead zone +-0.05 inside AckermannCar)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        randomise_start: bool = True,
        randomise_gap:   bool = False,
        render_mode            = None,
    ):
        super().__init__()

        self.randomise_start = randomise_start
        self.randomise_gap   = randomise_gap

        self._client, self._sim = connect()
        sim = self._sim

        self.ego   = AckermannCar(sim, EGO_NAME)
        self.park1 = AckermannCar(sim, PARK1_NAME)
        self.park2 = AckermannCar(sim, PARK2_NAME)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,),  dtype=np.float32)

        self._step_count = 0
        self._prev_dist  = None
        self._target     = np.array([TARGET_X, TARGET_Y], dtype=np.float32)
        self._target_yaw = TARGET_YAW

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sim = self._sim
        sim.stopSimulation()
        time.sleep(0.2)
        sim.startSimulation()
        time.sleep(0.2)

        if self.randomise_gap:
            self._randomise_gap()

        ego_x, ego_y, ego_yaw = (
            self._sample_start() if self.randomise_start
            else (TARGET_X - 1.2, TARGET_Y, 0.0)
        )
        self.ego.set_pose(ego_x, ego_y, ego_yaw)
        self.ego.stop()

        # Take one sim step so physics settles before we read obs
        self._client.step()

        self._step_count = 0
        obs = self._get_obs()
        self._prev_dist = _dist_from_obs(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        self.ego.set_controls(float(action[0]), float(action[1]))

        # KEY FIX: advance sim by exactly one dt
        self._client.step()
        self._step_count += 1

        obs                       = self._get_obs()
        reward, terminated, info  = self._compute_reward(obs)
        truncated                 = (self._step_count >= MAX_STEPS)

        if terminated or truncated:
            self.ego.stop()

        self._prev_dist = _dist_from_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._sim.stopSimulation()
        except Exception:
            pass

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        from coppeliasim_car import MAX_WHEEL_SPEED

        ego_pos, ego_yaw = self.ego.get_pose()
        ego_vel          = self.ego.get_velocity()
        p1_pos,  _       = self.park1.get_pose()
        p2_pos,  _       = self.park2.get_pose()

        t   = self._target
        yaw_err = _angle_wrap(self._target_yaw - ego_yaw)

        d_target = np.linalg.norm(t - ego_pos)
        dir_t    = _unit_vec(t - ego_pos)

        d_p1   = np.linalg.norm(p1_pos - ego_pos)
        dir_p1 = _unit_vec(p1_pos - ego_pos)
        d_p2   = np.linalg.norm(p2_pos - ego_pos)
        dir_p2 = _unit_vec(p2_pos - ego_pos)

        obs = np.array([
            d_target / WORLD_SCALE,          # 0
            dir_t[0], dir_t[1],              # 1-2
            math.cos(yaw_err),               # 3
            math.sin(yaw_err),               # 4
            ego_vel[0] / MAX_WHEEL_SPEED,    # 5
            ego_vel[1] / MAX_WHEEL_SPEED,    # 6
            ego_vel[2] / math.pi,            # 7
            dir_p1[0], dir_p1[1],            # 8-9
            d_p1 / WORLD_SCALE,              # 10
            dir_p2[0], dir_p2[1],            # 11-12
            d_p2 / WORLD_SCALE,              # 13
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, obs: np.ndarray):
        ego_pos, ego_yaw = self.ego.get_pose()
        p1_pos,  _       = self.park1.get_pose()
        p2_pos,  _       = self.park2.get_pose()

        dist     = _dist_from_obs(obs)
        yaw_err  = abs(_angle_wrap(self._target_yaw - ego_yaw))

        col1 = float(np.linalg.norm(ego_pos - p1_pos))
        col2 = float(np.linalg.norm(ego_pos - p2_pos))
        colliding = (col1 < COLLISION_DIST) or (col2 < COLLISION_DIST)

        r_proximity = math.exp(-3.0 * dist)
        r_progress  = ((self._prev_dist - dist) * 8.0) if self._prev_dist is not None else 0.0
        r_align     = math.exp(-3.0 * dist) * math.cos(yaw_err) * 0.5
        r_col       = -2.0 if colliding else 0.0
        r_time      = -0.002

        reward = r_proximity + r_progress + r_align + r_col + r_time

        terminated = False
        info = {"dist": dist, "yaw_err": math.degrees(yaw_err)}

        if dist < POS_THRESHOLD and yaw_err < YAW_THRESHOLD:
            reward    += 20.0
            terminated = True
            info["result"] = "success"
        elif colliding:
            reward    -= 5.0
            terminated = True
            info["result"] = "collision"

        return reward, terminated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_start(self):
        rng = self.np_random
        for _ in range(100):
            x   = rng.uniform(TARGET_X - 2.0, TARGET_X + 0.3)
            y   = rng.uniform(TARGET_Y - 1.0, TARGET_Y + 1.0)
            yaw = rng.uniform(-math.radians(45), math.radians(45))

            p1, _ = self.park1.get_pose()
            p2, _ = self.park2.get_pose()
            if (math.hypot(x - p1[0], y - p1[1]) > 0.5 and
                    math.hypot(x - p2[0], y - p2[1]) > 0.5):
                return x, y, yaw

        return TARGET_X - 1.2, TARGET_Y, 0.0

    def _randomise_gap(self):
        rng = self.np_random
        p1, yaw1 = self.park1.get_pose()
        p2, yaw2 = self.park2.get_pose()
        self.park1.set_pose(p1[0], p1[1] + rng.uniform(-0.15, 0.15), yaw1)
        self.park2.set_pose(p2[0], p2[1] + rng.uniform(-0.15, 0.15), yaw2)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)

def _dist_from_obs(obs: np.ndarray) -> float:
    return float(obs[0]) * WORLD_SCALE