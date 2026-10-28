"""
Quick sanity checks for gripper_base.py / rigid_gripper.py / soft_gripper.py.
Run directly: python test_grippers.py

Not a formal test suite (same spirit as test_arm_model.py) -- proof that the
two gripper models actually reproduce the behaviour your thesis is
predicting, before you wire them into the RL environment.
"""

import numpy as np

from gripper_base import GraspableObject, GripStatus
from rigid_gripper import RigidGripper
from soft_gripper import SoftGripper


def run_grasp(gripper, obj, object_position, ee_position, commanded_force, steps=200, dt=0.01):
    """Helper: hold a grasp attempt at a fixed alignment for `steps` ticks
    and command a constant grip force, then report the final status."""
    gripper.set_target_grip_force(commanded_force)
    for _ in range(steps):
        gripper.attempt_grasp(obj, object_position, ee_position, ee_velocity=np.zeros(3))
        gripper.step(dt)
    return gripper.status


def test_rigid_gripper_holds_a_normal_object():
    cube = GraspableObject(name="cube", mass_kg=0.3, radius_m=0.05,
                            fragility_n=1e6, friction_coeff=0.8)
    gripper = RigidGripper()
    status = run_grasp(gripper, cube,
                        object_position=[0, 0, 0], ee_position=[0, 0, 0],
                        commanded_force=20.0)
    print(f"[Rigid, cube, aligned] status={status.value}, grip_force={gripper.grip_force:.2f} N")
    assert status == GripStatus.HOLDING


def test_rigid_gripper_crushes_fragile_object():
    # An egg surrogate: light, but very low fragility limit.
    egg = GraspableObject(name="egg_surrogate", mass_kg=0.06, radius_m=0.03,
                           fragility_n=5.0, friction_coeff=0.5)
    gripper = RigidGripper()
    # Required force to hold it: (0.06 * 3.71) / 0.5 = 0.445 N -- comfortably
    # under the 5 N fragility limit, so a *careless* policy commanding too
    # much force is what crushes it here, exactly like a badly-tuned
    # standard-RL policy might.
    status = run_grasp(gripper, egg,
                        object_position=[0, 0, 0], ee_position=[0, 0, 0],
                        commanded_force=20.0)
    print(f"[Rigid, egg, overpowered] status={status.value}, grip_force={gripper.grip_force:.2f} N")
    assert status == GripStatus.CRUSHED


def test_rigid_gripper_fails_when_misaligned():
    cube = GraspableObject(name="cube", mass_kg=0.3, radius_m=0.05, friction_coeff=0.8)
    gripper = RigidGripper(alignment_tolerance_m=0.01)
    status = run_grasp(gripper, cube,
                        object_position=[0, 0, 0], ee_position=[0.03, 0, 0],  # 3 cm off
                        commanded_force=20.0)
    print(f"[Rigid, cube, misaligned 3cm] status={status.value}")
    assert status == GripStatus.NO_CONTACT


def test_soft_gripper_holds_fragile_object_where_rigid_crushes_it():
    egg = GraspableObject(name="egg_surrogate", mass_kg=0.06, radius_m=0.03,
                           fragility_n=5.0, friction_coeff=0.5)

    rigid = RigidGripper()
    soft = SoftGripper(num_pads=6)

    # Same total commanded force, same object, same alignment: 15 N is
    # enough to hold the egg (required ~0.45 N) but concentrated at two
    # rigid contact points it exceeds the 5 N fragility limit.
    rigid_status = run_grasp(rigid, egg, [0, 0, 0], [0, 0, 0], commanded_force=15.0)
    soft_status = run_grasp(soft, egg, [0, 0, 0], [0, 0, 0], commanded_force=15.0)

    print(f"[Rigid, egg, 15N total] status={rigid_status.value}")
    print(f"[Soft,  egg, 15N total] status={soft_status.value}, "
          f"peak_pad_force={soft.pad_forces.max():.2f} N")

    assert rigid_status == GripStatus.CRUSHED
    assert soft_status == GripStatus.HOLDING


def test_soft_gripper_tolerates_misalignment_where_rigid_fails():
    cube = GraspableObject(name="cube", mass_kg=0.3, radius_m=0.05, friction_coeff=0.8)

    rigid = RigidGripper(alignment_tolerance_m=0.01)
    soft = SoftGripper(num_pads=6, max_pad_deformation_m=0.02)

    misalignment_m = 0.015  # 1.5 cm: outside rigid's tolerance, inside soft's pad travel
    rigid_status = run_grasp(rigid, cube, [0, 0, 0], [misalignment_m, 0, 0], commanded_force=20.0)
    soft_status = run_grasp(soft, cube, [0, 0, 0], [misalignment_m, 0, 0], commanded_force=20.0)

    print(f"[Rigid, cube, 1.5cm off] status={rigid_status.value}")
    print(f"[Soft,  cube, 1.5cm off] status={soft_status.value}, "
          f"active_pads={np.count_nonzero(soft.pad_forces > 1e-6)}")

    assert rigid_status == GripStatus.NO_CONTACT
    assert soft_status == GripStatus.HOLDING


if __name__ == "__main__":
    test_rigid_gripper_holds_a_normal_object()
    test_rigid_gripper_crushes_fragile_object()
    test_rigid_gripper_fails_when_misaligned()
    test_soft_gripper_holds_fragile_object_where_rigid_crushes_it()
    test_soft_gripper_tolerates_misalignment_where_rigid_fails()
    print("\nAll gripper checks passed.")