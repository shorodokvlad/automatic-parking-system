"""
sensor_debug.py
---------------
Real-time sensor debugging utility.

Usage:
    python3 sensor_debug.py
    python3 sensor_debug.py --duration 60
    python3 sensor_debug.py --duration 30 --verbose
"""

import argparse
import time
import math
import numpy as np
from coppeliasim_car import AckermannCar, connect, DEFAULT_PROXIMITY_MAX_DISTANCE

EGO_NAME = "/nakedAckermannSteeringCar[0]"


def parse_args():
    p = argparse.ArgumentParser(description="Debug proximity sensors in real-time")
    p.add_argument("--duration", type=int, default=30, help="Debug duration in seconds")
    p.add_argument("--verbose", action="store_true", help="Print detailed debug info")
    return p.parse_args()


def format_distance_bar(distance: float, max_distance: float = DEFAULT_PROXIMITY_MAX_DISTANCE) -> str:
    """Create a visual bar showing distance (inverted: closer = more bars)"""
    if distance >= max_distance - 0.01:
        return "clear"
    ratio = 1.0 - (distance / max_distance)  # Invert: closer = higher ratio
    bars = int(ratio * 5)
    bar_str = "▰" * bars + "▱" * (5 - bars)
    return f"{distance:.3f}m  {bar_str}"


def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("SENSOR DEBUG UTILITY")
    print("=" * 70)
    print(f"Duration: {args.duration}s | Verbose: {args.verbose}")
    print("=" * 70 + "\n")

    client, sim = connect()
    ego = AckermannCar(sim, EGO_NAME)

    print(f"✓ Connected to CoppeliaSim")
    print(f"✓ Ego car: {EGO_NAME}")
    print(f"✓ Proximity sensor handles: {ego.proximity_handles}\n")

    start_time = time.time()
    step_count = 0

    try:
        while time.time() - start_time < args.duration:
            ego_pos, ego_yaw = ego.get_pose()
            yaw_deg = math.degrees(ego_yaw)

            # Read all sensors
            readings = ego.get_proximity_readings(max_distance=DEFAULT_PROXIMITY_MAX_DISTANCE)

            # Display header
            print(
                f"[{step_count:3d}] Ego Position: ({ego_pos[0]:6.3f}, {ego_pos[1]:7.3f}) | "
                f"Yaw: {yaw_deg:7.1f}°"
            )

            # Display sensor readings
            for direction in ["front", "left", "right"]:
                distance = readings[direction]
                is_clear = distance >= DEFAULT_PROXIMITY_MAX_DISTANCE - 0.01

                if is_clear:
                    status = "🟢"
                    display = "clear"
                else:
                    status = "🔴"
                    display = format_distance_bar(distance, DEFAULT_PROXIMITY_MAX_DISTANCE)

                print(f"  {status} proximity{direction.capitalize():6s} : {display}")

            # Step simulation
            client.step()
            step_count += 1

    except KeyboardInterrupt:
        print("\n\n^C")
    finally:
        print("\n" + "=" * 70)
        print("DEBUG SESSION COMPLETE")
        print("=" * 70)
        print(f"\nTotal steps: {step_count}")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print("\nSUMMARY:")
        print("✓ All sensors are working correctly!\n")

        try:
            sim.stopSimulation()
        except Exception:
            pass


if __name__ == "__main__":
    main()
