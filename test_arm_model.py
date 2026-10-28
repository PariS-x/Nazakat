"""
Quick sanity checks for arm_model.py. Run directly: python test_arm_model.py
Not a formal test suite — just proof the math holds together before you
build the gripper on top of it.
"""

import numpy as np
from arm_model import RoboticArm


def test_forward_kinematics_reaches_reasonable_workspace():
    arm = RoboticArm()
    arm.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
    ee = arm.end_effector_position_local()
    print("FK at all-zero pose:", ee)
    # Fully extended along x with base_height on z: rough sanity bound
    reach = sum(arm.link_lengths[1:])
    assert np.linalg.norm(ee[:2]) <= reach + 1e-6


def test_ik_round_trip():
    arm = RoboticArm()
    # Pick a target we know is reachable: run FK from a known pose, then
    # try to recover that same pose via IK from a different starting point.
    arm.joint_angles = np.array([0.3, -0.4, -0.9, 0.2])
    target = arm.end_effector_position_local()

    arm.joint_angles = np.zeros(4)  # different starting pose
    converged = arm.inverse_kinematics(target)
    arm.joint_angles = arm.target_angles.copy()
    achieved = arm.end_effector_position_local()

    error = np.linalg.norm(achieved - target)
    print(f"IK converged={converged}, position error={error:.5f} m")
    assert error < 5e-3


def test_pd_control_settles():
    arm = RoboticArm()
    arm.set_joint_targets([0.5, -0.3, -1.0, 0.4])

    dt = 0.01
    for _ in range(2000):  # 20 seconds sim time, plenty to settle
        arm.step(dt)

    error = np.linalg.norm(arm.joint_angles - arm.target_angles)
    print(f"Final joint error after settling: {error:.5f} rad")
    assert error < 1e-2


def test_world_position_with_rover_pose():
    arm = RoboticArm()
    arm.joint_angles = np.array([0.0, -0.2, -0.5, 0.1])

    # Rover at origin, no yaw
    p0 = arm.end_effector_position_world(rover_x=0.0, rover_y=0.0, rover_yaw=0.0)

    # Rover translated
    p1 = arm.end_effector_position_world(rover_x=5.0, rover_y=2.0, rover_yaw=0.0)
    assert np.allclose(p1 - p0, [5.0, 2.0, 0.0], atol=1e-6)

    # Rover rotated 90 degrees: local x becomes world y
    p2 = arm.end_effector_position_world(rover_x=0.0, rover_y=0.0, rover_yaw=np.pi / 2)
    print("EE world pos at 0 yaw:", p0, " | at 90 deg yaw:", p2)


if __name__ == "__main__":
    test_forward_kinematics_reaches_reasonable_workspace()
    test_ik_round_trip()
    test_pd_control_settles()
    test_world_position_with_rover_pose()
    print("\nAll arm_model checks passed.")