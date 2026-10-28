"""
Sanity checks for naz_env.py. Run directly: python test_naz_env.py

Same spirit as test_arm_model.py / test_grippers.py / test_manipulation_platform.py:
not a formal test suite, proof that the RL environment wraps the platform
correctly before you point Stable-Baselines3 at it.
"""

import numpy as np
from naz_env import NazManipulationEnv, OBS_DIM, ACT_DIM, _DEFAULT_JOINT_LIMITS


def _reset(env, seed=None):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, (terminated or truncated), info
    return result


def test_all_conditions_and_stages_run_without_crashing():
    """Every A/B/C/D condition, every curriculum stage, random actions."""
    for gripper_type in ("rigid", "soft"):
        env = NazManipulationEnv(gripper_type=gripper_type, mode="standard")
        obs = _reset(env, seed=0)
        assert obs.shape == (OBS_DIM,)
        for _ in range(30):
            obs, reward, done, info = _step(env, env.action_space.sample())
            if done:
                _reset(env)

        for stage in range(1, 7):
            env = NazManipulationEnv(gripper_type=gripper_type, mode="developmental", stage=stage)
            obs = _reset(env, seed=0)
            for _ in range(30):
                obs, reward, done, info = _step(env, env.action_space.sample())
                assert np.isfinite(reward), f"non-finite reward at stage {stage}"
                if done:
                    _reset(env)
    print("All conditions x stages ran without crashing, all rewards finite.")


def test_action_and_observation_shapes_are_fixed():
    """Shapes must not depend on gripper_type/mode/stage -- that's what lets
    a policy trained at one stage keep training at the next (see the
    curriculum-transfer discussion in naz_env.py's module docstring)."""
    shapes = set()
    for gripper_type in ("rigid", "soft"):
        for mode, stage in [("standard", 1), ("developmental", 1), ("developmental", 6)]:
            env = NazManipulationEnv(gripper_type=gripper_type, mode=mode, stage=stage)
            obs = _reset(env, seed=0)
            shapes.add((obs.shape, env.action_space.shape))
    assert len(shapes) == 1, f"observation/action shapes are not consistent: {shapes}"
    print(f"Observation/action shapes consistent across every condition: {shapes.pop()}")


def test_crushing_ends_the_episode():
    """A policy that slams full grip force onto a fragile object should
    terminate the episode via CRUSHED (not silently continue for 500 steps
    wasting rollout budget on a dead episode)."""
    env = NazManipulationEnv(gripper_type="rigid", mode="developmental", stage=4,
                              object_names=["cube"], episode_length=500)
    _reset(env, seed=0)
    # override the spawned object with a fragile one directly, then align
    # the end effector on top of it and squeeze at max force.
    from gripper_base import GraspableObject
    egg = GraspableObject(name="egg", mass_kg=0.06, radius_m=0.03, fragility_n=5.0, friction_coeff=0.5)
    env.platform.objects.clear()
    env.platform.spawn_object(egg, [0.4, 0.0, 0.3])
    env._object_name = "egg"
    env._object_spawn_pos = np.array([0.4, 0.0, 0.3])
    env._prev_object_pos = env._object_spawn_pos.copy()

    local_target = env.platform.ee_world_to_arm_local_target([0.4, 0.0, 0.3])
    env.platform.arm.inverse_kinematics(local_target)

    def normalize(targets):
        a = np.zeros(4)
        for i, (lo, hi) in enumerate(_DEFAULT_JOINT_LIMITS):
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            a[i] = np.clip((targets[i] - mid) / half, -1, 1)
        return a

    joint_action = normalize(env.platform.arm.target_angles)
    done = False
    steps_taken = 0
    for _ in range(300):
        obs, reward, done, info = _step(env, np.concatenate([joint_action, [1.0]]))
        steps_taken += 1
        if done:
            break

    print(f"Crush test: episode ended after {steps_taken} steps, "
          f"final status={info['gripper_status']}")
    assert done, "episode should have terminated on CRUSHED"
    assert info["gripper_status"] == "crushed"


def test_gripper_drop_recovery_is_reachable_through_the_env():
    """End-to-end check (through the actual RL-facing API, not the gripper
    class directly) that the DROPPED-state fix in rigid_gripper.py /
    soft_gripper.py is visible to a policy: commanding low then high grip
    force should be able to regain HOLDING within an episode."""
    env = NazManipulationEnv(gripper_type="rigid", mode="developmental", stage=4,
                              object_names=["cube"], episode_length=500)
    _reset(env, seed=0)
    object_pos = env.platform.object_position(env._object_name)
    local_target = env.platform.ee_world_to_arm_local_target(object_pos)
    env.platform.arm.inverse_kinematics(local_target)

    def normalize(targets):
        a = np.zeros(4)
        for i, (lo, hi) in enumerate(_DEFAULT_JOINT_LIMITS):
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            a[i] = np.clip((targets[i] - mid) / half, -1, 1)
        return a

    joint_action = normalize(env.platform.arm.target_angles)

    # grip hard, hold, release, then re-grip
    for grip_frac, n_steps in [(1.0, 100), (-1.0, 100), (1.0, 150)]:
        for _ in range(n_steps):
            obs, reward, done, info = _step(env, np.concatenate([joint_action, [grip_frac]]))
            if done:
                break

    print(f"Drop/recovery test final status: {info['gripper_status']}")
    assert info["gripper_status"] == "holding", (
        "gripper failed to recover HOLDING after being commanded to release and re-grip -- "
        "the DROPPED dead-end bug may have regressed"
    )


if __name__ == "__main__":
    test_all_conditions_and_stages_run_without_crashing()
    test_action_and_observation_shapes_are_fixed()
    test_crushing_ends_the_episode()
    test_gripper_drop_recovery_is_reachable_through_the_env()
    print("\nAll naz_env checks passed.")