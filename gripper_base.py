"""
gripper_base.py

Common interface for Project Naz's grippers (Gripper A: Rigid, Gripper B: Soft).

Why this file exists:
  Your 2x2 experiment (Rigid vs Soft) x (Standard vs Biomimetic RL) only stays
  clean if the RL loop, the arm controller, and the simulation engine can treat
  "the gripper" as one interchangeable thing. If RigidGripper and SoftGripper
  expose the same methods with the same meaning, you swap grippers with one
  line of config instead of branching logic through your whole codebase.

  This follows the same philosophy as DynamicRover and RoboticArm: plain
  Python classes, explicit state, a step(dt) method, nothing hidden.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

MARS_GRAVITY = 3.71  # m/s^2, matches dynamic_rover.MARS_GRAVITY


class GripStatus(Enum):
    """
    What the gripper is doing right now. Your developmental-curriculum reward
    functions (Stage 3 onward: Grip Discovery, Force Learning) should read
    this directly instead of re-deriving grasp success from raw forces.
    """
    IDLE = "idle"                # jaws/pads open, nothing happening
    APPROACHING = "approaching"  # closing on an object, force still ramping
    HOLDING = "holding"          # stably holding the object
    CRUSHED = "crushed"          # grip force exceeded the object's fragility limit
    DROPPED = "dropped"          # grip force too low to counter gravity -> object fell
    NO_CONTACT = "no_contact"    # too misaligned / out of reach to grip at all


@dataclass
class GraspableObject:
    """
    Minimal physical description of something the gripper might pick up.
    Deliberately generic so the same class covers both your training set
    (cube, ball, cylinder) and your test set (egg surrogate, sponge, smooth
    bottle, rock, etc) -- you just instantiate it with different numbers.

    mass_kg:        object mass.
    radius_m:       approximate distance from the object's center to its
                    surface, used for contact checks. For a cube use half
                    the side length -- it's an approximation, and that's
                    fine at this stage of the project.
    fragility_n:    maximum force (Newtons) the object can take at a single
                    contact point before it's damaged. Use a large number
                    (e.g. 1e6) for effectively-unbreakable objects (metal
                    cube), and a small number for fragile ones (egg
                    surrogate, paper cup).
    friction_coeff: friction between gripper surface and object surface.
                    Higher = easier to hold with less clamping force.
    """
    name: str
    mass_kg: float
    radius_m: float
    fragility_n: float = 1e6
    friction_coeff: float = 0.8


class BaseGripper(ABC):
    """
    Abstract interface every gripper variant must implement.

    Usage pattern, once per simulation timestep:

        gripper.attempt_grasp(obj, object_position, ee_position, ee_velocity)
        gripper.step(dt)
        status = gripper.status

    attempt_grasp() tells the gripper "here is an object at this world
    position, try to close on it, given where the end effector currently is".
    step() advances the gripper's own physics (force build-up, aperture
    motion) by dt. These are kept separate on purpose: your RL policy will
    call attempt_grasp() every step once it decides to grasp, but the force
    ramp needs its own time integration so Stage 4 (Force Learning) sees
    force rise gradually instead of snapping to a final value in one frame.
    """

    def __init__(self, max_grip_force: float = 40.0, closing_speed: float = 0.15):
        self.max_grip_force = max_grip_force   # Newtons, hardware/motor limit
        self.closing_speed = closing_speed     # m/s, how fast jaws/pads close

        self.status: GripStatus = GripStatus.IDLE
        self.grip_force: float = 0.0                       # current total clamping force, N
        self.target_grip_force: float = 0.0                 # what the RL policy is commanding
        self.target_object: Optional[GraspableObject] = None
        self.held_object: Optional[GraspableObject] = None

    @abstractmethod
    def attempt_grasp(self, obj: GraspableObject, object_position, ee_position, ee_velocity):
        """
        Try to close on `obj`, which is sitting at world position
        `object_position` (x, y, z). ee_position / ee_velocity come straight
        from RoboticArm.end_effector_position_world() and its derivative, so
        the gripper can judge alignment and impact speed.

        Must update self.status. Does not advance time itself -- call
        step(dt) separately.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float):
        """Advance internal gripper physics (force ramp, contact checks) by
        dt. Must update self.status."""
        raise NotImplementedError

    def set_target_grip_force(self, force: float):
        """RL policy calls this to command how hard to squeeze. Clamped to
        the gripper's own hardware limit."""
        self.target_grip_force = float(np.clip(force, 0.0, self.max_grip_force))

    def release(self):
        """Open the gripper and drop whatever it's holding."""
        self.status = GripStatus.IDLE
        self.grip_force = 0.0
        self.target_grip_force = 0.0
        self.held_object = None
        self.target_object = None

    def is_holding(self) -> bool:
        return self.status == GripStatus.HOLDING

    def get_state(self):
        """Snapshot for logging / RL observations, same pattern as
        RoboticArm.get_state()."""
        return {
            "status": self.status.value,
            "grip_force": self.grip_force,
            "target_grip_force": self.target_grip_force,
            "held_object": self.held_object.name if self.held_object else None,
        }