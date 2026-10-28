"""
env_wrappers.py

ActionRepeatWrapper -- repeats each policy action for `repeat` consecutive
physics steps of the underlying env before asking the policy for a new one.

WHY THIS EXISTS (debugging notes, kept here since the reason isn't obvious)
-----------------------------------------------------------------------------
NazManipulationEnv's action space sets joint TARGET angles directly (not an
IK-solved end-effector target), and RoboticArm's PD controller is rate
limited (max_joint_speed = 1.5 rad/s). At dt=0.01s, a single physics step
lets a joint move at most ~0.015 rad -- nowhere near enough to reach a
freshly-sampled target before the NEXT policy decision (every 0.01s) replaces
it with a different one. The arm never actually gets anywhere; it just
jitters around whatever pose it started at.

Measured directly: 400 episodes (200,000 steps) of literal uniform-random
exploration in the raw env never once got the end effector within 5cm of the
object (closest approach across all episodes: 3.4cm), and RigidGripper's
alignment tolerance is 1cm. That's not a "hasn't trained yet" symptom --
every one of those 400 episodes scored the exact same reward (-5.0, i.e.
-0.01 x 500 steps, zero variance), meaning SAC (or any off-policy method)
receives literally zero informative gradient no matter how long you train.
This is what caused Condition A/B eval to come back at a flat 0%.

Holding each sampled action for ~40-70 physics steps (instead of resampling
every single step) gives the PD controller enough time to actually approach
the commanded target, which is what starts producing real touch/hold events
during exploration -- confirmed empirically before writing this file.

This wrapper does NOT change reward semantics, reward magnitudes per
physics-tick, or the meaning of any action value -- it only changes how
often the policy is re-queried (same idea as frame-skip in classic DQN work,
a completely standard technique). Apply it uniformly across all four A/B/C/D
conditions so the comparison stays fair -- the control frequency is not
something that should differ between the standard and developmental legs of
the experiment.
"""

try:
    import gymnasium as gym
    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False
    gym = None


if _HAS_GYMNASIUM:

    class ActionRepeatWrapper(gym.Wrapper):
        """Repeat each action for `repeat` consecutive env.step() calls,
        summing reward and stopping early if the episode ends mid-repeat."""

        def __init__(self, env, repeat=50):
            super().__init__(env)
            if repeat < 1:
                raise ValueError(f"repeat must be >= 1, got {repeat}")
            self.repeat = repeat

        def step(self, action):
            total_reward = 0.0
            terminated = truncated = False
            obs, info = None, {}
            for _ in range(self.repeat):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            return obs, total_reward, terminated, truncated, info

else:
    # Sandbox fallback (gymnasium not installed) -- same idea, 4-tuple API,
    # mirrors naz_env.py's own _HAS_GYMNASIUM shim so this file stays
    # importable/testable without the real dependency.
    class ActionRepeatWrapper:
        def __init__(self, env, repeat=50):
            if repeat < 1:
                raise ValueError(f"repeat must be >= 1, got {repeat}")
            self.env = env
            self.repeat = repeat
            self.action_space = env.action_space
            self.observation_space = env.observation_space

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            total_reward = 0.0
            done = False
            info = {}
            obs = None
            for _ in range(self.repeat):
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
            return obs, total_reward, done, info
