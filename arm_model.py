"""
arm_model.py

Robotic arm module for Project Naz.

Provides a 4-DOF serial-link arm (base yaw, shoulder pitch, elbow pitch,
wrist pitch) that mounts onto the DynamicRover chassis. This is the piece
that sits between the rover physics engine (already built) and the
gripper (rigid / soft, still to come).

Design choices, and why:
  - DH parameters for forward kinematics: standard, well-tested, and it
    generalises cleanly if you add a 5th/6th joint later (e.g. wrist roll)
    without rewriting everything.
  - Numerical Jacobian instead of an analytical one: with 4 joints and
    changing link lengths during development, a numerical Jacobian is far
    less error-prone than re-deriving partial derivatives by hand every
    time you tweak a link length. It's slightly slower, but at this scale
    (4 joints, called once per control step) that cost is irrelevant.
  - Damped least-squares IK: plain pseudo-inverse IK blows up near
    singularities (e.g. arm fully extended). Damping trades a little
    accuracy for guaranteed stability, which matters once the base is
    also tilting under it (Stage 6 of your curriculum).
  - PD control on joint angles, integrated the same way DynamicRover
    integrates rover velocity: this keeps the two subsystems consistent
    when you step them together in the same simulation loop.
"""

import numpy as np


def dh_transform(theta, d, a, alpha):
    """Standard Denavit-Hartenberg homogeneous transform for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ])


class RoboticArm:
    """
    A 4-joint serial arm: base yaw -> shoulder pitch -> elbow pitch -> wrist pitch.

    link_lengths = (base_height, upper_arm, forearm, wrist_to_ee), all in metres.
    mount_offset = position of the arm's base joint relative to the rover's
                   chassis-center origin, in the rover's own frame (x forward,
                   y left, z up).
    """

    def __init__(self,
                 link_lengths=(0.15, 0.25, 0.25, 0.12),
                 joint_limits=None,
                 max_joint_speed=1.5,
                 mount_offset=(0.0, 0.0, 0.2)):

        self.link_lengths = link_lengths
        self.n_joints = 4

        self.joint_angles = np.zeros(self.n_joints)
        self.joint_velocities = np.zeros(self.n_joints)
        self.target_angles = np.zeros(self.n_joints)

        if joint_limits is None:
            joint_limits = [
                (-np.pi, np.pi),        # base yaw: full rotation
                (-np.pi / 2, np.pi / 2),  # shoulder pitch
                (-2.5, 0.2),              # elbow: mostly bends one way
                (-np.pi / 2, np.pi / 2),  # wrist pitch
            ]
        self.joint_limits = joint_limits

        self.max_joint_speed = max_joint_speed  # rad/s, per joint

        self.mount_offset = np.array(mount_offset, dtype=float)

        # PD gains, one pair per joint. Tuned loosely for a light arm;
        # you'll want to retune once real link masses/motors are known.
        self.kp = np.array([40.0, 60.0, 50.0, 20.0])
        self.kd = np.array([4.0, 6.0, 5.0, 2.0])

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def _dh_params(self):
        L0, L1, L2, L3 = self.link_lengths
        q = self.joint_angles
        # (theta, d, a, alpha) per joint
        return [
            (q[0], L0, 0.0, np.pi / 2),
            (q[1], 0.0, L1, 0.0),
            (q[2], 0.0, L2, 0.0),
            (q[3], 0.0, L3, 0.0),
        ]

    def forward_kinematics(self):
        """4x4 homogeneous transform of the end-effector, in the arm's own base frame."""
        T = np.eye(4)
        for theta, d, a, alpha in self._dh_params():
            T = T @ dh_transform(theta, d, a, alpha)
        return T

    def end_effector_position_local(self):
        """End-effector (x, y, z) in the arm's own base frame (no rover pose applied)."""
        return self.forward_kinematics()[:3, 3]

    def end_effector_position_world(self, rover_x=0.0, rover_y=0.0, rover_yaw=0.0):
        """
        End-effector position in world coordinates, given the rover's current
        pose. Assumes the arm base rotates rigidly with the rover's yaw and
        sits at rover_position + mount_offset (mount_offset already accounts
        for height above the chassis).
        """
        ee_local = self.end_effector_position_local() + self.mount_offset

        cy, sy = np.cos(rover_yaw), np.sin(rover_yaw)
        Rz = np.array([[cy, -sy, 0.0],
                        [sy,  cy, 0.0],
                        [0.0, 0.0, 1.0]])

        ee_world = Rz @ ee_local + np.array([rover_x, rover_y, 0.0])
        return ee_world

    def jacobian(self, delta=1e-4):
        """Numerical Jacobian (3xN) of end-effector position wrt joint angles."""
        J = np.zeros((3, self.n_joints))
        base = self.end_effector_position_local()

        for i in range(self.n_joints):
            original = self.joint_angles[i]
            self.joint_angles[i] = original + delta
            perturbed = self.end_effector_position_local()
            self.joint_angles[i] = original
            J[:, i] = (perturbed - base) / delta

        return J

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def set_joint_targets(self, targets):
        """Set target joint angles directly, clamped to joint limits."""
        targets = np.array(targets, dtype=float)
        for i, (lo, hi) in enumerate(self.joint_limits):
            targets[i] = np.clip(targets[i], lo, hi)
        self.target_angles = targets

    def inverse_kinematics(self, target_pos_local, max_iters=150, tol=1e-3, step_scale=0.5, damping=0.05):
        """
        Damped least-squares IK. target_pos_local is an (x, y, z) point in the
        arm's own base frame (i.e. before mount_offset / rover pose).

        On success, calls set_joint_targets() with the solved angles and
        returns True. On failure to converge, still sets the best-effort
        angles found and returns False, so the caller can decide whether
        to retry, back off, or flag an unreachable target.
        """
        q = self.joint_angles.copy()
        target = np.array(target_pos_local, dtype=float)
        saved_angles = self.joint_angles.copy()

        converged = False
        for _ in range(max_iters):
            self.joint_angles = q
            current = self.end_effector_position_local()
            error = target - current

            if np.linalg.norm(error) < tol:
                converged = True
                break

            J = self.jacobian()
            JT = J.T
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
            dq = JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(3), error)
            q = q + step_scale * dq

        self.joint_angles = saved_angles  # don't mutate actual state during solving
        self.set_joint_targets(q)
        return converged

    def step(self, dt):
        """
        Advance one timestep: PD control drives joint_velocities toward
        closing the gap to target_angles, then integrate position. Mirrors
        how DynamicRover integrates velocity -> position, so the two can be
        stepped together in the same loop without surprises.
        """
        error = self.target_angles - self.joint_angles
        control_signal = self.kp * error - self.kd * self.joint_velocities

        self.joint_velocities += control_signal * dt
        self.joint_velocities = np.clip(
            self.joint_velocities, -self.max_joint_speed, self.max_joint_speed
        )

        self.joint_angles += self.joint_velocities * dt

        for i, (lo, hi) in enumerate(self.joint_limits):
            if self.joint_angles[i] < lo:
                self.joint_angles[i] = lo
                self.joint_velocities[i] = 0.0
            elif self.joint_angles[i] > hi:
                self.joint_angles[i] = hi
                self.joint_velocities[i] = 0.0

    def reset(self):
        self.joint_angles[:] = 0.0
        self.joint_velocities[:] = 0.0
        self.target_angles[:] = 0.0

    def get_state(self):
        """Convenience snapshot, useful for logging / RL observations later."""
        return {
            "joint_angles": self.joint_angles.copy(),
            "joint_velocities": self.joint_velocities.copy(),
            "ee_position_local": self.end_effector_position_local(),
        }