"""
Sanity checks for manipulation_platform.py. Run directly:
    python test_manipulation_platform.py

Same spirit as test_arm_model.py / test_grippers.py: not a formal test
suite, but proof that rover + arm + gripper actually behave correctly
TOGETHER, not just individually. This is the first point in the project
where that's even possible to test.
"""

import numpy as np
from manipulation_platform import build_platform
from gripper_base import GraspableObject


def settle_arm_to(platform, world_target, dt=0.01, max_steps=1000, tol=0.01):
    """Drive the arm toward a world-frame point and step until it's close
    enough, or give up after max_steps. Returns True if it got there."""
    converged = platform.reach_toward(world_target)
    if not converged:
        print(f"  (warning) IK did not converge for target {world_target}")

    for _ in range(max_steps):
        platform.step(dt)
        error = np.linalg.norm(platform.ee_world_position() - np.array(world_target))
        if error < tol:
            return True
    return False


def test_arm_reaches_a_stationary_object():
    """With the rover parked, the arm should be able to reach and IK-solve
    toward an object placed within its workspace."""
    platform = build_platform(gripper_type="rigid")

    cube = GraspableObject(name="cube", mass_kg=0.3, radius_m=0.05, friction_coeff=0.8)
    # Something clearly inside the arm's reach (max reach ~ 0.15+0.25+0.25+0.12 = 0.77m)
    object_world_pos = [0.5, 0.0, 0.3]
    platform.spawn_object(cube, object_world_pos)

    reached = settle_arm_to(platform, object_world_pos)
    ee = platform.ee_world_position()
    error = np.linalg.norm(ee - np.array(object_world_pos))
    print(f"[reach test] reached={reached}, final EE={ee}, error={error:.4f} m")
    assert error < 0.02


def test_rigid_gripper_grasps_and_carries_object_when_rover_drives():
    """This is the actual point of this file: pick something up, then drive
    the rover, and confirm the object moves WITH the arm instead of staying
    frozen in its spawn position."""
    platform = build_platform(gripper_type="rigid")

    cube = GraspableObject(name="cube", mass_kg=0.3, radius_m=0.05, friction_coeff=0.8)
    object_world_pos = [0.5, 0.0, 0.3]
    platform.spawn_object(cube, object_world_pos)

    settle_arm_to(platform, object_world_pos)

    # Command a grasp for a while, ramping force up.
    for _ in range(300):
        platform.step(0.01, grasp_command=("cube", 30.0))

    status = platform.gripper.status.value
    print(f"[carry test] gripper status after grasp attempt: {status}")
    assert platform.gripper.is_holding(), f"expected HOLDING, got {status}"

    pos_before_drive = platform.object_position("cube").copy()

    # Now drive the rover forward for a bit (flat ground, no terrain -> pitch/roll stay 0).
    for _ in range(200):
        platform.step(0.01, drive_force=150, grasp_command=("cube", 30.0))

    pos_after_drive = platform.object_position("cube").copy()
    rover_moved = platform.rover.x > 0.05
    object_moved_with_rover = np.linalg.norm(pos_after_drive - pos_before_drive) > 0.05

    print(f"[carry test] rover.x={platform.rover.x:.3f}, "
          f"object moved by {np.linalg.norm(pos_after_drive - pos_before_drive):.3f} m, "
          f"still holding={platform.gripper.is_holding()}")

    assert rover_moved, "rover should have actually driven forward"
    assert object_moved_with_rover, "held object should move along with the rover+arm"
    assert platform.gripper.is_holding(), "object should still be held after driving"


def test_soft_gripper_survives_where_rigid_crushes_during_a_drive():
    """Reproduces the thesis claim (soft > rigid on fragile objects) but now
    inside the full platform, including the rover actually driving under
    the arm -- i.e. Stage 6 'Environment Adaptation' conditions, not just a
    static gripper test."""
    egg_specs = dict(name="egg", mass_kg=0.06, radius_m=0.03, fragility_n=5.0,
                      friction_coeff=0.5)

    results = {}
    for gripper_type in ("rigid", "soft"):
        platform = build_platform(gripper_type=gripper_type)
        egg = GraspableObject(**egg_specs)
        object_world_pos = [0.5, 0.0, 0.3]
        platform.spawn_object(egg, object_world_pos)

        settle_arm_to(platform, object_world_pos)

        # First, grip while stationary (this is when a real policy would
        # actually acquire the grasp -- you don't try to close a hand on a
        # target while your base is sliding out from under it). A slightly
        # careless, too-strong grip command (15N total, same as the earlier
        # gripper-only test).
        for _ in range(300):
            platform.step(0.01, grasp_command=("egg", 15.0))

        acquired_status = platform.gripper.status.value

        # NOW start driving, to see whether the grasp (rigid vs soft)
        # survives the disturbance of a moving/bumpy base -- this is the
        # actual Stage 6 "Environment Adaptation" scenario.
        for _ in range(300):
            platform.step(0.01, drive_force=100, grasp_command=("egg", 15.0))

        results[gripper_type] = platform.gripper.status.value
        print(f"  ({gripper_type}: status right after grip = {acquired_status}, "
              f"status after driving = {results[gripper_type]})")
        print(f"[fragile-under-drive test] {gripper_type} gripper -> {results[gripper_type]}")

    assert results["rigid"] == "crushed"
    assert results["soft"] == "holding"


if __name__ == "__main__":
    test_arm_reaches_a_stationary_object()
    test_rigid_gripper_grasps_and_carries_object_when_rover_drives()
    test_soft_gripper_survives_where_rigid_crushes_during_a_drive()
    print("\nAll manipulation_platform checks passed.")