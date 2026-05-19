"""
coppeliasim_car.py  (v2 — fixed stepping)
------------------------------------------
Key fix: synchronous stepping uses client.setStepping(True) + client.step().
Using sim.step() does NOT advance the simulation in the ZMQ API — the sim
runs freely, causing observations to "jump" (the "teleport" effect).
"""

import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

STEERING_JOINTS = ["steeringRight", "steeringLeft"]
MOTOR_JOINTS    = ["motorRight",    "motorLeft"]

MAX_STEER_ANGLE = math.radians(30)   # tune to your scene's joint limits
MAX_WHEEL_SPEED = 3.0                # rad/s — reduced for safer learning


class AckermannCar:
    def __init__(self, sim, base_name: str):
        self.sim  = sim
        self.name = base_name

        self.body_handle   = sim.getObject(base_name)
        self.steer_handles = [sim.getObject(f"{base_name}/{j}") for j in STEERING_JOINTS]
        self.motor_handles = [sim.getObject(f"{base_name}/{j}") for j in MOTOR_JOINTS]

    def set_controls(self, steer_norm: float, speed_norm: float):
        """
        steer_norm, speed_norm in [-1, 1].
        Dead zone ±0.05 on speed so the car can actually stop.
        """
        steer = float(np.clip(steer_norm, -1, 1)) * MAX_STEER_ANGLE
        raw   = float(np.clip(speed_norm, -1, 1))
        spd   = 0.0 if abs(raw) < 0.05 else raw * MAX_WHEEL_SPEED

        for h in self.steer_handles:
            self.sim.setJointTargetPosition(h, steer)
        for h in self.motor_handles:
            self.sim.setJointTargetVelocity(h, spd)

    def stop(self):
        for h in self.steer_handles:
            self.sim.setJointTargetPosition(h, 0.0)
        for h in self.motor_handles:
            self.sim.setJointTargetVelocity(h, 0.0)

    def get_pose(self) -> tuple[np.ndarray, float]:
        """Returns (xy_position, yaw_rad) in world frame."""
        pos = self.sim.getObjectPosition(self.body_handle, -1)
        eul = self.sim.getObjectOrientation(self.body_handle, -1)
        return np.array([pos[0], pos[1]], dtype=np.float32), float(eul[2])

    def get_velocity(self) -> np.ndarray:
        """Returns [vx, vy, omega_z] in world frame."""
        lin, ang = self.sim.getObjectVelocity(self.body_handle)
        return np.array([lin[0], lin[1], ang[2]], dtype=np.float32)

    def set_pose(self, x: float, y: float, yaw: float):
        sim = self.sim
        pos = sim.getObjectPosition(self.body_handle, -1)
        sim.setObjectPosition(self.body_handle, -1, [x, y, pos[2]])
        eul = sim.getObjectOrientation(self.body_handle, -1)
        sim.setObjectOrientation(self.body_handle, -1, [eul[0], eul[1], yaw])


def connect(host: str = "localhost", port: int = 23000):
    """
    Connect and enable synchronous stepping.
    IMPORTANT: keep the returned client alive — it owns the step() call.
    """
    client = RemoteAPIClient(host=host, port=port)
    sim    = client.require("sim")
    # ── Critical fix: enable stepping on the CLIENT, not via Lua ──────────
    client.setStepping(True)
    return client, sim