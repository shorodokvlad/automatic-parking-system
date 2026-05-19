"""
coppeliasim_car.py
------------------
Low-level wrapper around the CoppeliaSim ZMQ Remote API.
Handles synchronous stepping and reading proximity sensors.
"""

import math
import warnings
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

STEERING_JOINTS = ["steeringRight", "steeringLeft"]
MOTOR_JOINTS    = ["motorRight",    "motorLeft"]

MAX_STEER_ANGLE = math.radians(30)
MAX_WHEEL_SPEED = 3.0                
DEFAULT_PROXIMITY_MAX_DISTANCE = 3.0

class AckermannCar:
    def __init__(self, sim, base_name: str, has_sensors: bool = False):
        self.sim  = sim
        self.name = base_name

        self.body_handle   = sim.getObject(base_name)
        self.steer_handles = [sim.getObject(f"{base_name}/{j}") for j in STEERING_JOINTS]
        self.motor_handles = [sim.getObject(f"{base_name}/{j}") for j in MOTOR_JOINTS]
        
        # Only look for sensors if this car is supposed to have them
        if has_sensors:
            self.proximity_handles = self._resolve_proximity_sensor_handles()
        else:
            self.proximity_handles = {}
    def set_controls(self, steer_norm: float, speed_norm: float):
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
        pos = self.sim.getObjectPosition(self.body_handle, -1)
        eul = self.sim.getObjectOrientation(self.body_handle, -1)
        return np.array([pos[0], pos[1]], dtype=np.float32), float(eul[2])

    def get_velocity(self) -> np.ndarray:
        lin, ang = self.sim.getObjectVelocity(self.body_handle)
        return np.array([lin[0], lin[1], ang[2]], dtype=np.float32)

    def set_pose(self, x: float, y: float, yaw: float):
        sim = self.sim
        pos = sim.getObjectPosition(self.body_handle, -1)
        sim.setObjectPosition(self.body_handle, -1, [x, y, pos[2]])
        eul = sim.getObjectOrientation(self.body_handle, -1)
        sim.setObjectOrientation(self.body_handle, -1, [eul[0], eul[1], yaw])

    def get_proximity_readings(self, max_distance: float = DEFAULT_PROXIMITY_MAX_DISTANCE) -> dict[str, float]:
        distances = {}
        for direction, handle in self.proximity_handles.items():
            distances[direction] = self._read_sensor_distance(handle, max_distance)
        return distances

    def _resolve_proximity_sensor_handles(self) -> dict[str, int | None]:
        sensor_aliases = {
            "front": ["proximityFront", "frontProximity", "proximitySensorFront", "sensorFront"],
            "left":  ["proximityLeft", "leftProximity", "proximitySensorLeft", "sensorLeft"],
            "right": ["proximityRight", "rightProximity", "proximitySensorRight", "sensorRight"],
        }
        handles = {}
        for direction, names in sensor_aliases.items():
            handles[direction] = None
            for candidate_name in names:
                try:
                    handles[direction] = self.sim.getObject(f"{self.name}/{candidate_name}")
                    break
                except Exception:
                    continue
            if handles[direction] is None:
                warnings.warn(
                    f"{self.name}: proximity sensor for '{direction}' not found; using max-distance.",
                    RuntimeWarning, stacklevel=2
                )
        return handles

    def _read_sensor_distance(self, handle: int | None, max_distance: float) -> float:
        if handle is None:
            return float(max_distance)
        try:
            result = self.sim.readProximitySensor(handle)
        except Exception:
            return float(max_distance)

        detected, distance = self._parse_proximity_result(result)
        if not detected or distance is None:
            return float(max_distance)
        return float(np.clip(distance, 0.0, max_distance))

    def _parse_proximity_result(self, result) -> tuple[bool, float | None]:
        if not isinstance(result, (list, tuple)) or len(result) == 0:
            return False, None
        if not bool(result[0]):
            return False, None

        distance = None
        if len(result) >= 5:
            if isinstance(result[1], (int, float)):
                distance = float(result[1])
            elif isinstance(result[2], (list, tuple)) and len(result[2]) >= 3:
                distance = float(np.linalg.norm(np.array(result[2][:3], dtype=np.float32)))
        elif len(result) >= 2 and isinstance(result[1], (int, float)):
            distance = float(result[1])
        elif len(result) >= 2 and isinstance(result[1], (list, tuple)):
            if len(result[1]) >= 3:
                distance = float(np.linalg.norm(np.array(result[1][:3], dtype=np.float32)))
        return True, distance

def connect(host: str = "localhost", port: int = 23000):
    client = RemoteAPIClient(host=host, port=port)
    sim    = client.require("sim")
    client.setStepping(True)
    return client, sim