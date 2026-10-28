import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
except ImportError:
    # Allows this file to be imported and unit-tested in environments where gymnasium isn't installed 
    _HAS_GYMNASIUM = False

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype) if shape is None else np.full(shape, low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype) if shape is None else np.full(shape, high, dtype=dtype)
            self.shape = self.low.shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class spaces:
        Box = _Box

    class gym:
        class Env:
            pass


from manipulation_platform import build_platform
from gripper_base import GraspableObject, GripStatus


# ----------------------------------------------------------------------
# Training object set (Section 5.2 of the brief): cube, ball, cylinder.
# "radius_m" is used by both grippers as "distance from center to surface"
# per gripper_base.GraspableObject's own docstring -- for the cube this is
# half the side length, exactly as that docstring recommends.
# ----------------------------------------------------------------------
TRAINING_OBJECTS = {
    "cube":     dict(mass_kg=0.30, radius_m=0.05, fragility_n=1e6, friction_coeff=0.8),
    "ball":     dict(mass_kg=0.20, radius_m=0.04, fragility_n=1e6, friction_coeff=0.6),
    "cylinder": dict(mass_kg=0.25, radius_m=0.045, fragility_n=1e6, friction_coeff=0.7),
}

# ----------------------------------------------------------------------
# Held-out test object set (Section 5.3, Table 3 of the brief): never seen
# during training, used only for the generalization evaluation in eval.py.
#
# Honest limitation: our contact model (gripper_base.GraspableObject) only
# has three physical knobs -- mass, an effective radius, fragility, and
# friction. It has no real notion of "irregular geometry" the way an actual
# rock or toy has. For Rock/Toy specifically, "irregularity" is approximated
# by giving them a friction coefficient below the training-set objects and a
# larger radius_m (a rougher stand-in for "harder to get a clean, centered
# grasp on"), NOT literal non-convex geometry. This is a real modeling
# simplification, not a bug -- worth stating plainly in the report rather
# than pretending the sim captures object shape. Everything else (fragility
# for Egg/Paper Cup, low friction for the bottle) maps onto the model's
# existing knobs directly and faithfully.
#
# "cube_unseen" / "cylinder_unseen" cover Table 3's "Rigid regular" rows --
# Table 3 reuses the shape *names* Cube/Cylinder, but the point is these
# specific instances (different size/mass than the training cube/cylinder)
# were never seen during training, which is what "unseen during training"
# in Section 5.3 actually requires. Same shape family, different instance.
# ----------------------------------------------------------------------
TEST_OBJECTS = {
    # Rigid regular (Table 3)
    "cube_unseen":     dict(mass_kg=0.45, radius_m=0.07,  fragility_n=1e6, friction_coeff=0.75),
    "cylinder_unseen": dict(mass_kg=0.18, radius_m=0.03,  fragility_n=1e6, friction_coeff=0.65),
    # Rigid irregular (Table 3) -- see limitation note above
    "rock":            dict(mass_kg=0.35, radius_m=0.06,  fragility_n=1e6, friction_coeff=0.45),
    "toy":             dict(mass_kg=0.12, radius_m=0.06,  fragility_n=1e6, friction_coeff=0.5),
    # Fragile (Table 3) -- egg_surrogate values match test_grippers.py exactly
    "egg_surrogate":   dict(mass_kg=0.06, radius_m=0.03,  fragility_n=5.0, friction_coeff=0.5),
    "paper_cup":       dict(mass_kg=0.015, radius_m=0.04, fragility_n=8.0, friction_coeff=0.4),
    # Deformable (Table 3) -- modeled as high-fragility (won't "crush" in our
    # binary fragility sense) but still requiring a real hold force
    "sponge":          dict(mass_kg=0.03, radius_m=0.05,  fragility_n=1e6, friction_coeff=0.9),
    "foam_ball":       dict(mass_kg=0.02, radius_m=0.05,  fragility_n=1e6, friction_coeff=0.85),
    # Slippery (Table 3)
    "smooth_bottle":   dict(mass_kg=0.15, radius_m=0.035, fragility_n=1e6, friction_coeff=0.15),
}

# Single combined lookup used by NazManipulationEnv.reset() -- keeping this
# as one dict (rather than an if/else "is this a training or test name")
# means passing a mix of training and test object names to `object_names`
# just works, which eval.py relies on when it wants both sets in one run.
ALL_OBJECTS = {**TRAINING_OBJECTS, **TEST_OBJECTS}

_GRIP_STATUS_ORDER = [
    GripStatus.IDLE, GripStatus.APPROACHING, GripStatus.HOLDING,
    GripStatus.CRUSHED, GripStatus.DROPPED, GripStatus.NO_CONTACT,
]
_GRIP_STATUS_INDEX = {s: i for i, s in enumerate(_GRIP_STATUS_ORDER)}

# Joint limits repeated here only for action rescaling -- the arm remains the
# single source of truth for what limits actually apply during simulation
# (RoboticArm.set_joint_targets clips again anyway, so this can't get out of
# sync in a way that breaks physics; at worst a policy tries to command past
# the true limits and gets silently clipped).
_DEFAULT_JOINT_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi / 2, np.pi / 2),
    (-2.5, 0.2),
    (-np.pi / 2, np.pi / 2),
]

OBS_DIM = 29
ACT_DIM = 5

# How close the end effector must be to the object to count as "in contact",
# used by Stages 2 and 3. Deliberately looser than a gripper's own alignment
# tolerance (RigidGripper defaults to 0.01m) -- this is "close enough to be
# exploring the object," not "close enough to grasp it."
CONTACT_RADIUS_M = 0.06

# Stage 4/5/6's per-tick force-overshoot penalty coefficient: reward -=
# OVERSHOOT_PENALTY_COEFF * (grip_force - required_force), applied whenever
# HOLDING. Originally 0.05 -- confirmed (Conditions C and D, independently)
# to create a degenerate optimum where never attempting a grasp is safer in
# expectation than attempting one with an uncalibrated force policy, since a
# single sustained hold under bad calibration can accumulate a penalty
# (observed as low as -530 in one episode) that dwarfs any reward available
# for succeeding. A single named constant here (rather than the literal
# 0.05 duplicated in both _reward_stage_4_force_learning and
# _reward_stage_5_dexterity) makes testing a softer value a one-line change
# with no risk of missing an occurrence.
OVERSHOOT_PENALTY_COEFF = 0.05


class NazManipulationEnv(gym.Env):
    """
    One Gymnasium environment, parameterized to cover all 4 experimental
    conditions (A/B/C/D) and all 6 developmental curriculum stages.

    Parameters
    ----------
    gripper_type : "rigid" or "soft"
        The hardware half of the 2x2 design (Table 2 of the brief).
    mode : "standard" or "developmental"
        The learning half of the 2x2 design.
          - "standard": one fixed grasp-and-hold task, no staging. This is
            Condition A/B's training regime (Section 4.2 of the brief).
          - "developmental": reward function depends on `stage`. This is
            Condition C/D's training regime (Section 4.3, Table 1).
    stage : int, 1-6
        Only used when mode="developmental". Selects which reward function
        from Table 1 is active. Ignored (and irrelevant) when mode="standard".
    episode_length : int
        Max steps per episode before a timeout truncation.
    dt : float
        Physics timestep, seconds. 0.01 matches every other file in this
        project (arm PD control, gripper force ramp, rover integration all
        assume this scale of dt).
    object_names : list[str] or None
        Which objects to sample from each episode, looked up in
        naz_env.ALL_OBJECTS (training + test objects combined). Defaults to
        the three training objects (cube, ball, cylinder) per Section 5.2 --
        pass TRAINING for training, or a list of names from TEST_OBJECTS
        (e.g. ["egg_surrogate", "smooth_bottle"]) for generalization
        evaluation. Training and test names can be mixed in one list if
        you want a run that samples from both.
    seed : int or None
    """

    metadata = {"render_modes": []}

    def __init__(self, gripper_type="rigid", mode="standard", stage=1,
                 episode_length=500, dt=0.01, object_names=None, seed=None,
                 overshoot_ramp_steps=None, overshoot_ramp_start=0.0):
        """
        overshoot_ramp_steps, overshoot_ramp_start : optional, Stage 4/5 only.
            The untested "gradual ramp" fix proposed in the paper's Discussion
            (Section V), kept separate from the flat OVERSHOOT_PENALTY_COEFF
            every currently-reported result was measured under. If
            overshoot_ramp_steps is set, the effective overshoot coefficient
            starts at overshoot_ramp_start and rises linearly to
            OVERSHOOT_PENALTY_COEFF over that many cumulative environment
            steps (counted across episodes, not reset at each reset() --
            ramping is a property of the whole Stage 4/5 training run, not
            of any single episode). Leave overshoot_ramp_steps as None (the
            default) to reproduce every existing reported result exactly:
            the flat coefficient from step one, unchanged.
        """
        super().__init__()
        self._overshoot_ramp_steps = overshoot_ramp_steps
        self._overshoot_ramp_start = overshoot_ramp_start
        self._cumulative_env_steps = 0

        if gripper_type not in ("rigid", "soft"):
            raise ValueError(f"gripper_type must be 'rigid' or 'soft', got {gripper_type!r}")
        if mode not in ("standard", "developmental"):
            raise ValueError(f"mode must be 'standard' or 'developmental', got {mode!r}")
        if mode == "developmental" and stage not in range(1, 7):
            raise ValueError(f"stage must be 1-6, got {stage!r}")

        self.gripper_type = gripper_type
        self.mode = mode
        self.stage = stage
        self.episode_length = episode_length
        self.dt = dt
        self.object_names = object_names or list(TRAINING_OBJECTS.keys())

        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Built once; reset() re-configures rather than rebuilding, so the
        # platform's identity (and any object-model warm state) is stable
        # across episodes the way a real SB3 vec-env expects.
        self.platform = build_platform(gripper_type=gripper_type)

        self._object_name = None
        self._object_spawn_pos = None
        self._prev_object_pos = None
        self._episode_step = 0

        # Stage 1 (Exploration) reward needs a per-episode set of "novel
        # states visited" -- a voxel grid over the end effector's LOCAL
        # position (local, not world, so it means the same thing regardless
        # of where the rover happens to be parked).
        self._voxel_size = 0.03  # 3cm bins
        self._visited_voxels = set()

        # Stage 2 (Contact) reward needs "contact diversity across the
        # object's surface" -- approximated as distinct angular bins (around
        # the object, in the arm's local-frame-relative direction) that the
        # end effector has touched this episode.
        self._n_contact_bins = 12
        self._contacted_bins = set()

        # Reward dispatch table. Doing this once in __init__ rather than an
        # if/elif chain in step() keeps adding a 7th stage (if this project
        # ever needs one) to a one-line change instead of a step() edit.
        self._stage_reward_fns = {
            1: self._reward_stage_1_exploration,
            2: self._reward_stage_2_contact,
            3: self._reward_stage_3_grip_discovery,
            4: self._reward_stage_4_force_learning,
            5: self._reward_stage_5_dexterity,
            6: self._reward_stage_6_environment_adaptation,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Rebuild rover+arm+gripper fresh each episode. This is deliberately
        # simple over "clever partial reset" -- an RL env's reset() being
        # slightly slower is never the bottleneck; a reset() that silently
        # carries over stale state (a half-crushed object, a tipped-over
        # rover) absolutely would be, and would be a nightmare to debug.
        self.platform = build_platform(gripper_type=self.gripper_type)

        self._episode_step = 0
        self._visited_voxels = set()
        self._contacted_bins = set()
        self._completed_this_episode = False  # used by Stage 5/6's one-off completion bonus

        # Spawn one training object within the arm's comfortable workspace.
        # Reach is link_lengths[1:] summed (~0.62m for the default arm) --
        # sample well inside that so IK has a solution and the arm doesn't
        # spend the whole episode at full extension.
        object_name = self._rng.choice(self.object_names)
        obj_kwargs = dict(ALL_OBJECTS[object_name])
        obj = GraspableObject(name=object_name, **obj_kwargs)

        radius = self._rng.uniform(0.30, 0.55)
        angle = self._rng.uniform(-np.pi / 3, np.pi / 3)  # stay in front of the arm
        height = self._rng.uniform(0.15, 0.35)
        spawn_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])

        self.platform.spawn_object(obj, spawn_pos)
        self._object_name = object_name
        self._object_spawn_pos = spawn_pos.copy()
        self._prev_object_pos = spawn_pos.copy()

        obs = self._get_obs()
        info = {"object_name": object_name}
        return (obs, info) if _HAS_GYMNASIUM else obs

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        joint_targets = self._unnormalize_joint_targets(action[0:4])
        grip_force = self._unnormalize_grip_force(action[4])

        drive_force, turn_radius = self._apply_disturbance()

        step_info = self.platform.step(
            self.dt,
            drive_force=drive_force,
            turn_radius=turn_radius,
            arm_targets=joint_targets,
            grasp_command=(self._object_name, grip_force),
        )

        reward = self._compute_reward()
        self._episode_step += 1
        self._cumulative_env_steps += 1

        terminated = self._is_terminal()
        truncated = self._episode_step >= self.episode_length

        obs = self._get_obs()
        info = {
            "gripper_status": self.platform.gripper.status.value,
            "grip_force": self.platform.gripper.grip_force,
            "object_position": self.platform.object_position(self._object_name),
            "step_info": step_info,
            # Whether the object is currently held. train.py wraps this env
            # in SB3's Monitor with info_keywords=("success",), so Monitor
            # records THIS step's value from the final step of each episode
            # into monitor.csv -- that per-episode success flag is what
            # eval.py's compute_convergence_speed() reads back out to answer
            # "how many episodes of training did it take to reach 80%
            # success", i.e. Section 5.4's "RL convergence speed" metric.
            "success": self.platform.gripper.status == GripStatus.HOLDING,
        }

        self._prev_object_pos = self.platform.object_position(self._object_name)

        if _HAS_GYMNASIUM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info

    # ------------------------------------------------------------------
    # Action (de)normalization
    # ------------------------------------------------------------------
    def _unnormalize_joint_targets(self, normalized):
        """Map action[0:4] in [-1, 1] to each joint's actual radian limits."""
        targets = np.zeros(4)
        for i, (lo, hi) in enumerate(_DEFAULT_JOINT_LIMITS):
            mid = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            targets[i] = mid + normalized[i] * half_range
        return targets

    def _unnormalize_grip_force(self, normalized):
        """Map action[4] in [-1, 1] to [0, max_grip_force]. Negative actions
        just mean 'don't grip' (clamped to 0), so the policy doesn't need to
        learn a dead zone around 0 -- any non-positive value already means
        the same thing."""
        frac = (normalized + 1.0) / 2.0  # [-1,1] -> [0,1]
        frac = max(0.0, min(1.0, frac))
        return frac * self.platform.gripper.max_grip_force

    # ------------------------------------------------------------------
    # Disturbance (Stage 6 only)
    # ------------------------------------------------------------------
    def _apply_disturbance(self):
        """
        Scripted rover motion for Stage 6 (Environment Adaptation). The
        policy never controls the rover directly (see module docstring for
        why) -- the environment drives it at a fixed, moderate force so the
        arm/gripper policy has to cope with a genuinely moving, tilting base,
        matching the brief's "manipulation on a moving rover" framing.

        Returns (drive_force, turn_radius), both None outside Stage 6 so
        ManipulationPlatform.step() leaves the rover's existing commanded
        force/turn radius untouched (it defaults to 0 / None from
        DynamicRover.__init__ and is never otherwise set in Stages 1-5).
        """
        if self.mode == "developmental" and self.stage == 6:
            return 100.0, None  # same magnitude used in test_manipulation_platform.py
        return None, None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        arm = self.platform.arm
        rover = self.platform.rover
        gripper = self.platform.gripper

        ee_world = self.platform.ee_world_position()
        object_pos = self.platform.object_position(self._object_name)
        if object_pos is None:
            object_pos = np.zeros(3)

        status_onehot = np.zeros(len(_GRIP_STATUS_ORDER), dtype=np.float32)
        status_onehot[_GRIP_STATUS_INDEX[gripper.status]] = 1.0

        obs = np.concatenate([
            arm.joint_angles.astype(np.float32),
            arm.joint_velocities.astype(np.float32),
            ee_world.astype(np.float32),
            object_pos.astype(np.float32),
            (object_pos - ee_world).astype(np.float32),
            np.array([gripper.grip_force], dtype=np.float32),
            status_onehot,
            np.array([rover.x, rover.y, rover.yaw, rover.velocity, rover.roll], dtype=np.float32),
        ])
        assert obs.shape == (OBS_DIM,), f"observation shape drifted: {obs.shape}"
        return obs

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _is_terminal(self):
        """
        End the episode early on an unrecoverable outcome. CRUSHED is a true
        dead end (gripper_base.py's own state machine treats it that way, on
        purpose -- see rigid_gripper.py's attempt_grasp docstring). Everything
        else (DROPPED, NO_CONTACT) is recoverable within the episode after
        the state-machine fix in rigid_gripper.py / soft_gripper.py, so it's
        left to the reward function to discourage rather than ending the
        episode over it.
        """
        return self.platform.gripper.status == GripStatus.CRUSHED

    # ------------------------------------------------------------------
    # Reward dispatch
    # ------------------------------------------------------------------
    def _compute_reward(self):
        if self.mode == "standard":
            return self._reward_standard()
        return self._stage_reward_fns[self.stage]()

    def _reward_standard(self):
        """
        Standard RL baseline (Section 4.2): "define a task (pick up the
        object) and a reward function (+1 for success, -1 for dropping),
        then train the policy directly." No developmental shaping -- this is
        deliberately the plain, un-staged reward that Conditions A and B
        train against, so any generalization gap seen against C/D is
        attributable to the curriculum, not to this reward being secretly
        richer.
        """
        gripper = self.platform.gripper
        reward = -0.01  # small per-step cost, encourages finishing quickly

        if gripper.status == GripStatus.HOLDING:
            reward += 1.0
        elif gripper.status == GripStatus.CRUSHED:
            reward -= 5.0
        elif gripper.status == GripStatus.DROPPED:
            reward -= 1.0

        return reward

    # ------------------------------------------------------------------
    # Developmental curriculum reward functions (Table 1 of the brief)
    # ------------------------------------------------------------------
    def _reward_stage_1_exploration(self):
        """
        Table 1, Stage 1: "Curiosity-driven reaching; reward = novel state
        visitation." Implemented as a voxel-grid novelty bonus over the end
        effector's LOCAL position: +1 the first time a 3cm voxel is visited
        this episode, 0 on repeat visits, minus a tiny per-step cost so
        standing still isn't a free lunch.
        """
        ee_local = self.platform.arm.end_effector_position_local()
        voxel = tuple(np.floor(ee_local / self._voxel_size).astype(int))

        if voxel not in self._visited_voxels:
            self._visited_voxels.add(voxel)
            return 1.0
        return -0.01

    def _reward_stage_2_contact(self):
        """
        Table 1, Stage 2: "Surface exploration; reward = contact diversity
        across object surface." Implemented as: when the end effector is
        within CONTACT_RADIUS_M of the object, bin the approach direction
        into one of 12 angular sectors (around the object, in the XY plane);
        +1 the first time a sector is touched this episode, small step cost
        otherwise. This rewards touching the object from many different
        sides rather than poking the same spot repeatedly.
        """
        ee_world = self.platform.ee_world_position()
        object_pos = self.platform.object_position(self._object_name)
        offset = ee_world - object_pos
        dist = np.linalg.norm(offset)

        if dist > CONTACT_RADIUS_M:
            return -0.01  # not in contact; small cost to keep exploring toward it

        angle = np.arctan2(offset[1], offset[0])
        bin_idx = int((angle + np.pi) / (2 * np.pi) * self._n_contact_bins) % self._n_contact_bins

        if bin_idx not in self._contacted_bins:
            self._contacted_bins.add(bin_idx)
            return 1.0
        return 0.05  # still in contact, just not a new bin -- mildly positive, not punished

    def _reward_stage_3_grip_discovery(self):
        """
        Table 1, Stage 3: "Learning grasp affordances; reward = object
        displacement from resting state." Implemented as the object's
        distance moved (this step) away from its original spawn position --
        an accidental or intentional grasp that drags the object registers
        as positive reward, exactly like an infant's accidental grasp
        producing an interesting, reinforcing effect on the world.
        """
        object_pos = self.platform.object_position(self._object_name)
        displacement_from_rest = np.linalg.norm(object_pos - self._object_spawn_pos)
        step_progress = np.linalg.norm(object_pos - self._prev_object_pos)

        reward = 2.0 * step_progress + 0.1 * displacement_from_rest
        if self.platform.gripper.status == GripStatus.CRUSHED:
            reward -= 5.0
        return reward

    def _current_overshoot_coeff(self):
        """
        Returns the overshoot penalty coefficient to use right now. If no
        ramp schedule was set (the default), this is always the flat
        OVERSHOOT_PENALTY_COEFF, exactly what every currently-reported
        result was measured under. If a ramp schedule was set (see
        __init__), this rises linearly from overshoot_ramp_start to
        OVERSHOOT_PENALTY_COEFF over overshoot_ramp_steps cumulative
        environment steps, then holds at OVERSHOOT_PENALTY_COEFF for the
        remainder of training.
        """
        if self._overshoot_ramp_steps is None:
            return OVERSHOOT_PENALTY_COEFF
        frac = min(1.0, self._cumulative_env_steps / self._overshoot_ramp_steps)
        return self._overshoot_ramp_start + frac * (OVERSHOOT_PENALTY_COEFF - self._overshoot_ramp_start)

    def _reward_stage_4_force_learning(self):
        """
        Table 1, Stage 4: "Adaptive grip force; reward = +object held -
        damage proxy (force overshoot)." Implemented literally: +1 per step
        held, minus a penalty proportional to how far grip_force overshoots
        the minimum force actually required to hold the object (the same
        `required_force = mass*g/friction` computation the grippers
        themselves use internally) -- i.e. squeezing harder than necessary
        is punished even before it crushes anything, which is what teaches
        the policy to find the minimum sufficient force rather than just
        maxing out grip force to be safe.
        """
        gripper = self.platform.gripper
        reward = 0.0

        if gripper.status == GripStatus.HOLDING:
            reward += 1.0
            obj = gripper.held_object
            if obj is not None:
                required = (obj.mass_kg * 3.71) / max(obj.friction_coeff, 1e-3)
                overshoot = max(0.0, gripper.grip_force - required)
                reward -= self._current_overshoot_coeff() * overshoot
        elif gripper.status == GripStatus.CRUSHED:
            reward -= 5.0
        elif gripper.status == GripStatus.DROPPED:
            reward -= 0.5

        return reward

    def _reward_stage_5_dexterity(self):
        """
        Table 1, Stage 5: "Fine manipulation; reward = task completion with
        minimal excess force." Same overshoot penalty as Stage 4, plus a
        larger one-off completion bonus the first time HOLDING is reached
        each episode (so the policy is pushed toward actually closing the
        loop -- reach, grasp, hold -- rather than just idling in a
        low-overshoot state that was never gripping anything).
        """
        gripper = self.platform.gripper
        reward = -0.02  # slightly larger time penalty than Stage 4: efficiency matters more here

        if gripper.status == GripStatus.HOLDING:
            if not getattr(self, "_completed_this_episode", False):
                reward += 5.0
                self._completed_this_episode = True
            reward += 0.5
            obj = gripper.held_object
            if obj is not None:
                required = (obj.mass_kg * 3.71) / max(obj.friction_coeff, 1e-3)
                overshoot = max(0.0, gripper.grip_force - required)
                reward -= self._current_overshoot_coeff() * overshoot
        elif gripper.status == GripStatus.CRUSHED:
            reward -= 5.0
        elif gripper.status == GripStatus.DROPPED:
            reward -= 1.0

        return reward

    def _reward_stage_6_environment_adaptation(self):
        """
        Table 1, Stage 6: "Manipulation on moving rover; reward = task
        success under platform disturbance." Same shaping as Stage 5 (it's
        the same underlying skill), since the disturbance itself is what
        makes this stage harder -- the reward doesn't need to separately
        reward "not being disturbed," the rover motion (see
        _apply_disturbance) does that work by making HOLDING genuinely
        difficult to sustain.
        """
        return self._reward_stage_5_dexterity()