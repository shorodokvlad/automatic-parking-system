"""
parking_env.py  (v3 — scene-independent sensor fusion)
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_car import (
    AckermannCar,
    connect,
    MAX_WHEEL_SPEED,
    DEFAULT_PROXIMITY_MAX_DISTANCE,
)

# ── Scene-independent object constants ────────────────────────────────────────
EGO_NAME = "/nakedAckermannSteeringCar[0]"

# ── Training constants ────────────────────────────────────────────────────────

MAX_STEPS      = 900
WORLD_SCALE    = 3.0
SENSOR_RANGE   = DEFAULT_PROXIMITY_MAX_DISTANCE

# Thresholds for success
POS_THRESHOLD   = 0.35
YAW_THRESHOLD   = math.radians(20)
BALANCE_THRESH  = 0.18
FRONT_SAFE_MIN  = 0.20
FRONT_SAFE_MAX  = 0.90
TARGET_FRONT_DISTANCE = 0.55
COLLISION_FRONT_THRESHOLD = 0.12
COLLISION_SIDE_THRESHOLD = 0.10

# Gap estimator constants (metres / scale factors), tuned as scene-agnostic defaults.
# Tuning basis: standard passenger-car parking dimensions and a ~3m short-range sensor setup.
# If sensor range or vehicle footprint changes significantly, recalibrate these values.
GAP_FORWARD_SCALE = 0.55
GAP_FORWARD_BIAS = 0.20
GAP_FORWARD_MIN = 0.35
GAP_FORWARD_MAX = 1.60
GAP_LATERAL_SCALE = 0.75
GAP_LATERAL_MIN = 0.30
GAP_LATERAL_MAX = 1.20

DIFFICULTY_CONFIG = {
    # max_steps: number of simulation steps per episode
    # sensor_noise: additive Gaussian noise in metres for proximity readings
    # start_radius: spawn randomization radius around scene ego-car anchor in metres
    # start_yaw_range: spawn randomization heading range in degrees
    # pos_threshold: success distance threshold in metres
    "easy":   {"max_steps": 700,  "sensor_noise": 0.00, "start_radius": 0.6, "start_yaw_range": 35.0, "pos_threshold": 0.40},
    "medium": {"max_steps": 900,  "sensor_noise": 0.01, "start_radius": 1.0, "start_yaw_range": 45.0, "pos_threshold": 0.35},
    "hard":   {"max_steps": 1200, "sensor_noise": 0.02, "start_radius": 1.4, "start_yaw_range": 55.0, "pos_threshold": 0.30},
}

OBS_IDX_FRONT = 0
OBS_IDX_LEFT = 1
OBS_IDX_RIGHT = 2
OBS_IDX_DIST = 3
OBS_IDX_LATERAL_BALANCE = 11


class ParallelParkingEnv(gym.Env):
    """
    Gymnasium env for parallel parking with nakedAckermannSteeringCar.

    Observation (12 floats):
      [0]    front proximity / SENSOR_RANGE              (0 -> 1)
      [1]    left  proximity / SENSOR_RANGE              (0 -> 1)
      [2]    right proximity / SENSOR_RANGE              (0 -> 1)
      [3]    dist to estimated gap center / WORLD_SCALE  (0 -> 1)
      [4-5]  unit vector ego->estimated gap center       (-1 -> 1)
      [6]    cos(yaw_error_to_gap)                       (-1 -> 1)
      [7]    sin(yaw_error_to_gap)                       (-1 -> 1)
      [8-9]  ego velocity (vx, vy) / MAX_WHEEL_SPEED     (-1 -> 1)
      [10]   ego angular velocity / pi                   (-1 -> 1)
      [11]   lateral balance (right-left)/SENSOR_RANGE   (-1 -> 1)

    Action (2 floats in [-1, 1]):
      [0]  steer_norm
      [1]  speed_norm  (dead zone +-0.05 inside AckermannCar)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        randomise_start: bool = True,
        difficulty: str = "medium",
        max_steps: int | None = None,
        render_mode            = None,
    ):
        super().__init__()

        self.randomise_start = randomise_start
        if difficulty not in DIFFICULTY_CONFIG:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. Expected one of {tuple(DIFFICULTY_CONFIG.keys())}."
            )
        if SENSOR_RANGE <= 0.0:
            raise ValueError(
                f"SENSOR_RANGE (from DEFAULT_PROXIMITY_MAX_DISTANCE) must be positive, got {SENSOR_RANGE}"
            )
        self.difficulty = difficulty

        cfg = DIFFICULTY_CONFIG[self.difficulty]
        self._max_steps      = int(max_steps if max_steps is not None else cfg["max_steps"])
        self._sensor_noise   = float(cfg["sensor_noise"])
        self._start_radius   = float(cfg["start_radius"])
        self._start_yaw_range = math.radians(float(cfg["start_yaw_range"]))
        self._pos_threshold  = float(cfg["pos_threshold"])

        self._client, self._sim = connect()
        sim = self._sim

        self.ego = AckermannCar(sim, EGO_NAME)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,),  dtype=np.float32)

        self._step_count = 0
        self._prev_dist  = None
        self._target     = np.zeros(2, dtype=np.float32)
        self._target_yaw = 0.0
        self._spawn_anchor = np.zeros(2, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sim = self._sim
        sim.stopSimulation()
        time.sleep(0.2)
        sim.startSimulation()
        time.sleep(0.2)

        anchor_pos, anchor_yaw = self.ego.get_pose()
        self._spawn_anchor = anchor_pos.copy()
        ego_x, ego_y, ego_yaw = (
            self._sample_start(anchor_pos, anchor_yaw) if self.randomise_start
            else (float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_yaw))
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
        truncated                 = (self._step_count >= self._max_steps)

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
        ego_pos, ego_yaw = self.ego.get_pose()
        ego_vel          = self.ego.get_velocity()
        prox             = self._read_proximity()
        t, t_yaw         = self._estimate_gap_pose(ego_pos, ego_yaw, prox)
        self._target = t
        self._target_yaw = t_yaw

        yaw_err = _angle_wrap(t_yaw - ego_yaw)
        d_target = np.linalg.norm(t - ego_pos)
        dir_t = _unit_vec(t - ego_pos)
        lateral_balance = (prox["right"] - prox["left"]) / SENSOR_RANGE

        obs = np.array([
            prox["front"] / SENSOR_RANGE,    # 0
            prox["left"] / SENSOR_RANGE,     # 1
            prox["right"] / SENSOR_RANGE,    # 2
            d_target / WORLD_SCALE,          # 3
            dir_t[0], dir_t[1],              # 4-5
            math.cos(yaw_err),               # 6
            math.sin(yaw_err),               # 7
            ego_vel[0] / MAX_WHEEL_SPEED,    # 8
            ego_vel[1] / MAX_WHEEL_SPEED,    # 9
            ego_vel[2] / math.pi,            # 10
            lateral_balance,                 # 11
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, obs: np.ndarray):
        _, ego_yaw = self.ego.get_pose()

        front    = float(obs[OBS_IDX_FRONT]) * SENSOR_RANGE
        left     = float(obs[OBS_IDX_LEFT]) * SENSOR_RANGE
        right    = float(obs[OBS_IDX_RIGHT]) * SENSOR_RANGE
        dist     = _dist_from_obs(obs)
        balance  = abs(float(obs[OBS_IDX_LATERAL_BALANCE]))
        yaw_err  = abs(_angle_wrap(self._target_yaw - ego_yaw))

        collision_risk = (
            (front < COLLISION_FRONT_THRESHOLD)
            or (left < COLLISION_SIDE_THRESHOLD)
            or (right < COLLISION_SIDE_THRESHOLD)
        )
        front_in_slot  = FRONT_SAFE_MIN <= front <= FRONT_SAFE_MAX

        r_proximity = math.exp(-2.5 * dist)
        r_progress  = ((self._prev_dist - dist) * 8.0) if self._prev_dist is not None else 0.0
        r_align     = math.exp(-2.0 * dist) * math.cos(yaw_err) * 0.6
        r_balance   = 0.5 * (1.0 - balance)
        r_front     = 0.3 if front_in_slot else -0.3 * abs(front - TARGET_FRONT_DISTANCE)
        r_col       = -2.5 if collision_risk else 0.0
        r_time      = -0.004

        reward = r_proximity + r_progress + r_align + r_balance + r_front + r_col + r_time

        terminated = False
        info = {"dist": dist, "yaw_err": math.degrees(yaw_err), "front": front, "balance": balance}

        is_position_aligned = dist < self._pos_threshold
        is_heading_aligned = yaw_err < YAW_THRESHOLD
        is_laterally_centered = balance < BALANCE_THRESH
        is_front_distance_valid = front_in_slot

        if is_position_aligned and is_heading_aligned and is_laterally_centered and is_front_distance_valid:
            reward    += 25.0
            terminated = True
            info["result"] = "success"
        elif collision_risk:
            reward    -= 6.0
            terminated = True
            info["result"] = "collision"

        return reward, terminated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_start(self, anchor_pos: np.ndarray, anchor_yaw: float):
        rng = self.np_random
        for _ in range(100):
            x = anchor_pos[0] + rng.uniform(-self._start_radius, self._start_radius)
            y = anchor_pos[1] + rng.uniform(-self._start_radius, self._start_radius)
            yaw = anchor_yaw + rng.uniform(-self._start_yaw_range, self._start_yaw_range)
            return float(x), float(y), float(yaw)

        return float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_yaw)

    def _read_proximity(self) -> dict[str, float]:
        readings = self.ego.get_proximity_readings(max_distance=SENSOR_RANGE)
        if self._sensor_noise > 0.0:
            for k, v in readings.items():
                noisy = v + float(self.np_random.normal(0.0, self._sensor_noise))
                readings[k] = float(np.clip(noisy, 0.0, SENSOR_RANGE))
        return readings

    def _estimate_gap_pose(
        self,
        ego_pos: np.ndarray,
        ego_yaw: float,
        prox: dict[str, float],
    ) -> tuple[np.ndarray, float]:
        forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)

        open_side = "left" if prox["left"] >= prox["right"] else "right"
        side_sign = 1.0 if open_side == "left" else -1.0

        max_side_distance = max(prox["left"], prox["right"])
        # Forward offset combines a small constant bias and a linear front-clearance term:
        # this biases the target into open space while still reacting to nearby obstacles.
        raw_forward_offset = prox["front"] * GAP_FORWARD_SCALE + GAP_FORWARD_BIAS
        forward_offset = float(np.clip(
            raw_forward_offset,
            GAP_FORWARD_MIN,
            GAP_FORWARD_MAX,
        ))
        lateral_offset = float(np.clip(
            max_side_distance * GAP_LATERAL_SCALE,
            GAP_LATERAL_MIN,
            GAP_LATERAL_MAX,
        ))

        target = ego_pos + forward * forward_offset + lateral * side_sign * lateral_offset
        target_dir = target - ego_pos
        target_yaw = float(math.atan2(target_dir[1], target_dir[0]))
        return target.astype(np.float32), target_yaw


# ── Utilities ─────────────────────────────────────────────────────────────────

def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)

def _dist_from_obs(obs: np.ndarray) -> float:
    return float(obs[OBS_IDX_DIST]) * WORLD_SCALE
