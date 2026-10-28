"""
manipulation_platform.py

THE MISSING LINK.

Right now you have three things that each work perfectly on their own:
  - DynamicRover      (drives around, tips over, drains battery)
  - RoboticArm         (reaches a target point, given in its own local frame)
  - RigidGripper/SoftGripper (opens/closes on an object, given world positions)

But nothing calls all three together, in the right order, every timestep.
That's what this file does. It is deliberately "dumb" -- it doesn't decide
WHAT to do (that's the RL policy's job, coming next). It only makes sure
that when something tells it "drive forward, move the arm here, squeeze
this hard on that object", all three subsystems update consistently and
a held object actually follows the gripper around.

Why this needs to exist as its own file (not just glued together inside
main.py): the RL environment we build next needs one clean object to
reset() and step(), the same way test_grippers.py needed one clean
run_grasp() helper instead of repeating gripper.attempt_grasp() /
gripper.step() everywhere. Same idea, one level up.

Coordinate frames, spelled out (this is where most bugs happen):
  - rover.x, rover.y, rover.yaw          -> world frame
  - arm.joint_angles                     -> arm's own local frame
  - arm.end_effector_position_local()    -> point in the ARM's local frame
  - arm.end_effector_position_world(...) -> that point converted into the
                                             WORLD frame, using the rover's
                                             CURRENT pose
  - object positions in this file        -> always WORLD frame
  - gripper.attempt_grasp(..., ee_position, ...) -> must be given WORLD
                                             frame positions, which is why
                                             we always pass the *_world()
                                             version, never the local one.

Getting local vs. world mixed up is the single most common bug when you
wire an arm onto a moving base, so every method below is named to make
which frame it's in explicit.
"""

import numpy as np

from gripper_base import GraspableObject, GripStatus


class ManipulationPlatform:
    """
    One object = one rover + one arm + one gripper + a table of objects
    sitting in the world. Call step() once per simulation tick.
    """

    def __init__(self, rover, arm, gripper):
        self.rover = rover
        self.arm = arm
        self.gripper = gripper

        # name -> {"obj": GraspableObject, "position": np.array([x, y, z])}
        # Positions are WORLD frame, always.
        self.objects = {}

        self.time = 0.0

        # Used to estimate end-effector velocity by finite difference, which
        # RigidGripper/SoftGripper accept but don't currently use for physics
        # -- kept here anyway so a future impact-speed rule has real data
        # instead of everyone silently passing zeros.
        self._prev_ee_world = self.ee_world_position()

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------
    def spawn_object(self, obj: GraspableObject, position):
        """Place an object at a WORLD-frame (x, y, z) position."""
        self.objects[obj.name] = {
            "obj": obj,
            "position": np.array(position, dtype=float),
        }

    def remove_object(self, name):
        self.objects.pop(name, None)

    def object_position(self, name):
        entry = self.objects.get(name)
        return entry["position"].copy() if entry else None

    # ------------------------------------------------------------------
    # Frame conversions
    # ------------------------------------------------------------------
    def ee_world_position(self):
        """End-effector position in WORLD frame, using the rover's current pose."""
        return self.arm.end_effector_position_world(
            rover_x=self.rover.x, rover_y=self.rover.y, rover_yaw=self.rover.yaw
        )

    def ee_world_to_arm_local_target(self, world_target):
        """
        Convert a WORLD-frame point (e.g. an object's position) into the
        point RoboticArm.inverse_kinematics() needs, which is expressed in
        the arm's own base frame (i.e. undo the rover's yaw/translation and
        the arm's mount_offset).

        This is what you call before doing IK toward an object: you know
        where the object is in the world, but arm.inverse_kinematics()
        only understands its own local frame.
        """
        world_target = np.array(world_target, dtype=float)

        # Undo rover translation
        relative = world_target - np.array([self.rover.x, self.rover.y, 0.0])

        # Undo rover yaw (rotate by -yaw)
        cy, sy = np.cos(-self.rover.yaw), np.sin(-self.rover.yaw)
        Rz_inv = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ])
        local = Rz_inv @ relative

        # Undo the arm's fixed mounting offset on the chassis
        local_to_arm_base = local - self.arm.mount_offset
        return local_to_arm_base

    # ------------------------------------------------------------------
    # High-level commands (what an RL policy or test script calls)
    # ------------------------------------------------------------------
    def reach_toward(self, world_target, max_iters=150):
        """
        Point the arm's IK at a WORLD-frame target and set joint targets
        accordingly. Does not move anything itself -- step() still has to
        be called to actually integrate the PD controller toward those
        targets. Returns True if IK converged (target is reachable).
        """
        local_target = self.ee_world_to_arm_local_target(world_target)
        return self.arm.inverse_kinematics(local_target, max_iters=max_iters)

    def command_grasp(self, object_name, target_grip_force):
        """
        Try to grasp `object_name` this step, commanding `target_grip_force`
        newtons of squeeze. Pass object_name=None to release whatever is
        currently held (or stop trying to grasp).
        """
        if object_name is None:
            self.gripper.release()
            return

        entry = self.objects.get(object_name)
        if entry is None:
            raise KeyError(
                f"No object named '{object_name}' in this platform. "
                f"Known objects: {list(self.objects.keys())}"
            )

        self.gripper.set_target_grip_force(target_grip_force)
        ee_pos = self.ee_world_position()
        ee_vel = (ee_pos - self._prev_ee_world)  # per-step displacement, not /dt on purpose:
        # attempt_grasp only uses this for judging impact softness in future
        # extensions; callers that need true velocity should divide by dt
        # themselves. Kept simple here to avoid a wrong implicit unit.
        self.gripper.attempt_grasp(entry["obj"], entry["position"], ee_pos, ee_vel)

    # ------------------------------------------------------------------
    # The one method everything else is built on
    # ------------------------------------------------------------------
    def step(self, dt, drive_force=None, turn_radius=None, arm_targets=None,
              grasp_command=None):
        """
        Advance rover, arm, and gripper by dt, in a fixed, deliberate order:

          1. rover moves first, because the arm's world position depends on
             the rover's pose, and we want the arm reacting to THIS step's
             rover pose, not last step's stale one.
          2. arm moves second, using (optionally) new joint targets.
          3. gripper moves third, using the arm's freshly-updated
             end-effector position -- so alignment checks reflect where the
             hand actually is *after* this step's motion, not before it.
          4. if the gripper is holding something, that object's world
             position is snapped to the end effector -- this is what makes
             a held object "come along for the ride" when the rover drives
             or the arm moves.

        Parameters are all optional so you can call step(dt) on its own to
        just let physics settle with no new commands.

        drive_force / turn_radius : forwarded straight to DynamicRover.
        arm_targets               : 4-vector of joint angles (radians) to
                                     aim the PD controller at this step.
        grasp_command             : (object_name_or_None, target_force)
        """
        if drive_force is not None:
            self.rover.set_drive_force(drive_force)
        if turn_radius is not None:
            self.rover.set_turn_radius(turn_radius)

        rover_state = self.rover.step(dt)

        if arm_targets is not None:
            self.arm.set_joint_targets(arm_targets)
        self.arm.step(dt)

        # Drag any object that was ALREADY held into the fresh end-effector
        # position now, before checking alignment for this step's grasp
        # command. This has to happen here, not after gripper.step() below --
        # otherwise a held object trails one whole timestep behind a moving
        # rover, and RigidGripper's alignment check (which re-checks every
        # step, not just at first contact) sees that lag as "the object
        # slipped 8cm away" on a fast-moving base and drops it. The object
        # is mechanically rigid with the gripper once held; it should never
        # appear to have moved relative to the hand just because the base
        # moved this step.
        if self.gripper.is_holding() and self.gripper.held_object is not None:
            held_name = self.gripper.held_object.name
            if held_name in self.objects:
                self.objects[held_name]["position"] = self.ee_world_position()

        if grasp_command is not None:
            object_name, target_force = grasp_command
            self.command_grasp(object_name, target_force)
        self.gripper.step(dt)

        self._prev_ee_world = self.ee_world_position()
        self.time += dt

        return {
            "time": round(self.time, 3),
            "rover": rover_state,
            "arm": self.arm.get_state(),
            "gripper": self.gripper.get_state(),
            "objects": {
                name: entry["position"].copy() for name, entry in self.objects.items()
            },
        }

    def reset_arm_and_gripper(self):
        """Reset arm to neutral pose and open the gripper, without touching
        the rover (useful between grasp attempts within the same episode)."""
        self.arm.reset()
        self.gripper.release()


# ----------------------------------------------------------------------
# Factory: builds one of the four experimental conditions (A/B/C/D) from
# your 2x2 table. This is intentionally the ONLY place that decides "rigid
# vs soft" -- everything downstream (the RL env, training scripts) just
# asks for a condition name and gets a correctly-wired platform back, so
# there's exactly one place a morphology bug could hide instead of four
# copy-pasted setups drifting apart from each other.
# ----------------------------------------------------------------------
def build_platform(gripper_type="rigid", rover_kwargs=None, arm_kwargs=None,
                    gripper_kwargs=None):
    """
    gripper_type: "rigid" or "soft". This is the hardware half of the 2x2
                  design (the learning-curriculum half is chosen later, by
                  which training script/reward function you run this
                  platform through -- it's not a property of the platform).
    """
    from dynamic_rover import DynamicRover
    from arm_model import RoboticArm
    from rigid_gripper import RigidGripper
    from soft_gripper import SoftGripper

    rover_kwargs = rover_kwargs or dict(
        mass=120, wheel_base=1.2, track_width=1.0, body_height=0.8
    )
    arm_kwargs = arm_kwargs or {}
    gripper_kwargs = gripper_kwargs or {}

    rover = DynamicRover(**rover_kwargs)
    arm = RoboticArm(**arm_kwargs)

    if gripper_type == "rigid":
        gripper = RigidGripper(**gripper_kwargs)
    elif gripper_type == "soft":
        gripper = SoftGripper(**gripper_kwargs)
    else:
        raise ValueError(f"gripper_type must be 'rigid' or 'soft', got {gripper_type!r}")

    return ManipulationPlatform(rover, arm, gripper)