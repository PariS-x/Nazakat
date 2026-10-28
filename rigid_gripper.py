"""
rigid_gripper.py

Gripper A from the Naz 2x2 experiment: servo-actuated hard jaws, fixed
closing shape, zero passive compliance. Everything the gripper "knows"
about how to hold an object has to come from the RL policy commanding
grip force precisely -- the hardware gives it no help.

Model used here, and why:

  - Two-point pinch, checked at a single effective contact: this is the
    standard simplification for parallel-jaw grippers when you're not
    modeling full 3D contact geometry (see Mahler et al. 2019, Dex-Net).
    You don't need finite-element contact for an RL-scale capstone
    project -- you need something fast enough to run thousands of
    episodes and physically honest enough that the results mean
    something.

  - Alignment tolerance: because there's no compliance, if the end
    effector is more than `alignment_tolerance_m` away from the
    object's center when it tries to close, the grasp fails outright --
    jaws close on air or catch the object off-center and it slips out.
    This is the concrete, testable way "rigid grippers don't forgive
    misalignment" (from your brief, Section 3.2) shows up in code.

  - Grip force vs. fragility: the force needed to hold the object is
    derived from basic Coulomb friction -- the clamping force, scaled
    by the friction coefficient, must be at least enough to support the
    object's weight: F_required = (m * g) / mu. If the force needed to
    reach that threshold exceeds the object's fragility limit, the
    object gets crushed before it can be held. This is exactly what
    should make Condition A (rigid + standard RL) fail on your fragile
    test objects (egg surrogate, paper cup) -- which is the whole point
    of running the experiment.
"""

import numpy as np

from gripper_base import BaseGripper, GripStatus, GraspableObject, MARS_GRAVITY


class RigidGripper(BaseGripper):

    def __init__(self,
                 max_grip_force: float = 40.0,
                 closing_speed: float = 0.15,
                 alignment_tolerance_m: float = 0.01,
                 force_ramp_rate: float = 80.0):
        """
        alignment_tolerance_m: how far off-center (meters) the end
            effector may be from the object and still get a stable grasp.
            Real parallel-jaw grippers are unforgiving here -- 1 cm is
            already generous for a small manipulator arm.
        force_ramp_rate: how fast grip force can rise, N/s. Not instant,
            because real servos take time to build torque, and because
            Stage 4 (Force Learning) needs to see the ramp, not a jump.
        """
        super().__init__(max_grip_force, closing_speed)
        self.alignment_tolerance_m = alignment_tolerance_m
        self.force_ramp_rate = force_ramp_rate

    # ------------------------------------------------------------------
    def attempt_grasp(self, obj: GraspableObject, object_position, ee_position, ee_velocity):
        # CRUSHED is a terminal state: a broken object doesn't un-break just
        # because the end effector later drifts away from where it broke.
        # Without this guard, every subsequent attempt_grasp() call (which
        # ManipulationPlatform fires every step the grasp command is active)
        # re-checks alignment against a static object position and, once the
        # rover has driven far enough away, overwrites CRUSHED with
        # NO_CONTACT -- silently erasing the crush event.
        if self.status == GripStatus.CRUSHED and self.target_object is obj:
            return False

        self.target_object = obj

        ee_position = np.asarray(ee_position, dtype=float)
        object_position = np.asarray(object_position, dtype=float)
        misalignment = np.linalg.norm(ee_position - object_position)

        if misalignment > self.alignment_tolerance_m:
            # Rigid jaws cannot compensate for this -- no contact is made,
            # and any force already built up bleeds off since the jaws
            # aren't actually touching anything.
            self.status = GripStatus.NO_CONTACT
            self.grip_force = 0.0
            self.held_object = None
            return False

        # IDLE / NO_CONTACT / DROPPED can all re-enter APPROACHING: only CRUSHED
        # is a true dead end (handled by the guard above). Without DROPPED here,
        # a policy that drops an object and immediately re-commands full grip
        # force at the same alignment can never regain HOLDING -- grip_force
        # stays frozen at whatever it was when the drop happened, because
        # step() only ramps force while status is APPROACHING or HOLDING.
        if self.status in (GripStatus.IDLE, GripStatus.NO_CONTACT, GripStatus.DROPPED):
            self.status = GripStatus.APPROACHING
        return True

    # ------------------------------------------------------------------
    def step(self, dt: float):
        if self.status not in (GripStatus.APPROACHING, GripStatus.HOLDING):
            return

        # Ramp grip force toward whatever the policy last commanded.
        error = self.target_grip_force - self.grip_force
        max_delta = self.force_ramp_rate * dt
        self.grip_force += float(np.clip(error, -max_delta, max_delta))
        self.grip_force = max(0.0, self.grip_force)

        obj = self.target_object if self.status == GripStatus.APPROACHING else self.held_object
        if obj is None:
            return

        # Crushed takes priority over holding -- you can crush an object
        # on the way to a force that would otherwise have held it fine.
        if self.grip_force >= obj.fragility_n:
            self.status = GripStatus.CRUSHED
            self.held_object = None
            return

        required_force = (obj.mass_kg * MARS_GRAVITY) / max(obj.friction_coeff, 1e-3)

        if self.status == GripStatus.APPROACHING:
            if self.grip_force >= required_force:
                self.status = GripStatus.HOLDING
                self.held_object = obj
        elif self.status == GripStatus.HOLDING:
            if self.grip_force < required_force:
                self.status = GripStatus.DROPPED
                self.held_object = None