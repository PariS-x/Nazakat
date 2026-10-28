"""
soft_gripper.py

Gripper B from the Naz 2x2 experiment: silicone-based, cephalopod-inspired,
passively compliant. No rigid jaw shape -- it wraps around whatever it
touches and distributes contact force across its surface instead of
pinching at two hard points.

Model used here, and why:

  A full FEM soft-body simulation would model exactly how the silicone
  deforms in 3D. That's the right tool if you were validating a physical
  material design. It is the wrong tool here: it's computationally
  expensive (bad for an RL loop that needs thousands of episodes/hour)
  and it answers a question your thesis isn't asking. Your research
  question is about morphology's effect on manipulation robustness, not
  about predicting exact silicone stress fields.

  Instead this models the gripper as `num_pads` independent contact
  pads arranged around the object (like suckers along a cephalopod arm
  segment). Each pad behaves like a damped spring: the more it's
  compressed against the object, the more force it exerts, capped by how
  far it can physically deform. This is a standard, cheap approximation
  for compliant contact (the same idea used in most soft-gripper RL sims,
  e.g. Graule et al. 2022's SoMoGym) and it reproduces the two properties
  that actually matter for your experiment:

  - Force distribution: total clamping force is spread across the pads
    that are in contact, so peak force *per contact point* stays much
    lower than a rigid two-point pinch applying the same total force.
    That's what should let Condition B survive fragile objects that
    Condition A crushes.
  - Misalignment tolerance: pads that are further from the object still
    deform and contribute *some* force (up to their max deformation),
    rather than contributing zero like a rigid jaw that's missed its
    mark. So a soft grasp degrades gracefully instead of failing outright.
"""

import numpy as np

from gripper_base import BaseGripper, GripStatus, GraspableObject, MARS_GRAVITY


class SoftGripper(BaseGripper):

    def __init__(self,
                 max_grip_force: float = 40.0,
                 closing_speed: float = 0.15,
                 num_pads: int = 6,
                 max_pad_deformation_m: float = 0.02,
                 pad_stiffness: float = 4000.0,
                 force_ramp_rate: float = 80.0):
        """
        num_pads: how many independent contact points are modeled around
            the gripper's wrap. More pads = smoother force distribution,
            at the cost of more per-step computation. 6 is a reasonable
            default for a first pass.
        max_pad_deformation_m: how far a single pad can compress before
            it's fully bottomed out. This is what gives the gripper its
            extended tolerance to misalignment -- a pad up to this far
            from the object can still make contact and push back.
        pad_stiffness: spring constant per pad (N/m). Higher = firmer
            material, builds force faster per meter of compression.
            You'll want to retune this once you have real silicone
            durometer / compression test data from the hardware team.
        """
        super().__init__(max_grip_force, closing_speed)
        self.num_pads = num_pads
        self.max_pad_deformation_m = max_pad_deformation_m
        self.pad_stiffness = pad_stiffness
        self.force_ramp_rate = force_ramp_rate

        # Per-pad state, indexed 0..num_pads-1. pad_positions holds each
        # pad's fixed angular offset around the gripper's closing axis --
        # this is what "distributed across the surface" means concretely.
        self.pad_angles = np.linspace(0, 2 * np.pi, num_pads, endpoint=False)
        self.pad_forces = np.zeros(num_pads)

    # ------------------------------------------------------------------
    def _pad_engagement(self, misalignment_m: float) -> np.ndarray:
        """
        How much each pad can compress, given how far off-center the end
        effector is. Pads are arranged in a ring, so a misalignment along
        one direction compresses pads on the near side more than pads on
        the far side -- this is a coarse but honest stand-in for "the
        gripper wraps unevenly when approached off-axis".

        Returns an array of available deformation per pad, in meters,
        already clipped to [0, max_pad_deformation_m].
        """
        # Pads "facing into" the misalignment direction get extra
        # effective reach; pads facing away get less. cos() gives a
        # smooth -1..1 spread across the ring; the offset re-centers it
        # to non-negative before we add the baseline deformation.
        directional_bias = np.cos(self.pad_angles) * misalignment_m
        engagement = self.max_pad_deformation_m - misalignment_m + directional_bias
        return np.clip(engagement, 0.0, self.max_pad_deformation_m)

    # ------------------------------------------------------------------
    def attempt_grasp(self, obj: GraspableObject, object_position, ee_position, ee_velocity):
        # Same terminal-state guard as RigidGripper: CRUSHED must stick.
        # See rigid_gripper.py's attempt_grasp() for the full explanation.
        if self.status == GripStatus.CRUSHED and self.target_object is obj:
            return False

        self.target_object = obj

        ee_position = np.asarray(ee_position, dtype=float)
        object_position = np.asarray(object_position, dtype=float)
        misalignment = np.linalg.norm(ee_position - object_position)

        max_reach = self.max_pad_deformation_m * 1.5  # a bit of slack beyond full pad travel
        if misalignment > max_reach:
            self.status = GripStatus.NO_CONTACT
            self.grip_force = 0.0
            self.pad_forces[:] = 0.0
            self.held_object = None
            return False

        # Same fix as RigidGripper.attempt_grasp: DROPPED must be re-enterable
        # into APPROACHING, or a policy that re-commands force after a drop can
        # never regain HOLDING. Only CRUSHED (guarded above) is a true dead end.
        if self.status in (GripStatus.IDLE, GripStatus.NO_CONTACT, GripStatus.DROPPED):
            self.status = GripStatus.APPROACHING
        self._last_misalignment = misalignment
        return True

    # ------------------------------------------------------------------
    def step(self, dt: float):
        if self.status not in (GripStatus.APPROACHING, GripStatus.HOLDING):
            return

        obj = self.target_object if self.status == GripStatus.APPROACHING else self.held_object
        if obj is None:
            return

        # Ramp the commanded total force the same way the rigid gripper
        # does, so both grippers are being asked to do "the same thing"
        # from the RL policy's point of view.
        error = self.target_grip_force - self.grip_force
        max_delta = self.force_ramp_rate * dt
        self.grip_force += float(np.clip(error, -max_delta, max_delta))
        self.grip_force = max(0.0, self.grip_force)

        # Distribute that total force across pads in proportion to how
        # much each pad is actually engaged (compressed). A pad with zero
        # available deformation contributes zero force, however hard the
        # gripper is squeezing overall.
        engagement = self._pad_engagement(getattr(self, "_last_misalignment", 0.0))
        weights = engagement * self.pad_stiffness
        total_weight = weights.sum()

        if total_weight <= 1e-9:
            self.pad_forces[:] = 0.0
        else:
            self.pad_forces = self.grip_force * (weights / total_weight)

        active_pads = np.count_nonzero(engagement > 1e-6)
        if active_pads == 0:
            self.status = GripStatus.NO_CONTACT
            self.held_object = None
            return

        peak_pad_force = float(self.pad_forces.max())

        # Crushed if *any single pad* exceeds the object's per-contact
        # fragility limit -- but because force is spread across pads
        # instead of two points, the same total grip_force produces a
        # much lower peak_pad_force than RigidGripper would.
        if peak_pad_force >= obj.fragility_n:
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

    def get_state(self):
        state = super().get_state()
        state["peak_pad_force"] = float(self.pad_forces.max()) if len(self.pad_forces) else 0.0
        state["active_pads"] = int(np.count_nonzero(self.pad_forces > 1e-6))
        return state