import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from naz_env import NazManipulationEnv, TRAINING_OBJECTS, TEST_OBJECTS, ALL_OBJECTS


# ----------------------------------------------------------------------
# Rollout: SB3-independent by design (see module docstring)
# ----------------------------------------------------------------------
def _step(env, action):
    """Normalize both the real gymnasium 5-tuple and this project's
    no-gymnasium-installed 4-tuple fallback (see naz_env.py's _HAS_GYMNASIUM
    shim) to one shape, so the rest of this file doesn't need to care which
    one is active."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, (terminated or truncated), info
    return result


def _reset(env, seed=None):
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def collect_episode_record(env, action_fn, dt, hold_threshold_seconds=3.0,
                            recovery_window_seconds=1.5, seed=None, action_repeat=1):
    """
    Run exactly one episode and return a plain-dict record of everything
    the metric functions below need. See module docstring for the metric
    definitions this feeds.

    action_repeat: re-invoke action_fn only every `action_repeat` raw env
    steps, reusing the last action otherwise -- matching whatever the policy
    was trained under (see env_wrappers.py / train.py's --action-repeat).
    The raw env is still stepped every physics tick (dt granularity), so
    hold_threshold_seconds / recovery_window_seconds accounting below is
    completely unaffected; this only changes how often action_fn is called,
    not the timing math.
    """
    obs = _reset(env, seed=seed)

    hold_threshold_steps = int(round(hold_threshold_seconds / dt))
    recovery_window_steps = int(round(recovery_window_seconds / dt))

    current_hold_streak = 0
    max_hold_streak = 0
    first_holding_step = None
    grip_forces_while_holding = []

    prev_status = "idle"
    drop_step = None          # step index of the most recent unresolved drop
    drop_events = []          # list of bool: True if that drop was recovered

    step_idx = 0
    done = False
    action = None
    while not done:
        if step_idx % action_repeat == 0:
            action = action_fn(obs)
        obs, reward, done, info = _step(env, action)
        status = info["gripper_status"]

        if status == "holding":
            current_hold_streak += 1
            max_hold_streak = max(max_hold_streak, current_hold_streak)
            if first_holding_step is None:
                first_holding_step = step_idx
            grip_forces_while_holding.append(info["grip_force"])

            # A drop that gets recovered is resolved the moment we're back
            # to HOLDING -- whether or not it was within the window is
            # decided right here, not after the episode ends.
            if drop_step is not None:
                recovered = (step_idx - drop_step) <= recovery_window_steps
                drop_events.append(recovered)
                drop_step = None
        else:
            current_hold_streak = 0

        if prev_status == "holding" and status == "dropped":
            drop_step = step_idx

        prev_status = status
        step_idx += 1

    # Any drop that was still unresolved when the episode ended counts as
    # not-recovered (there was no more time left for the policy to try).
    if drop_step is not None:
        drop_events.append(False)

    had_crush = (prev_status == "crushed")
    reached_holding = first_holding_step is not None
    hold_success = max_hold_streak >= hold_threshold_steps

    return {
        "object_name": env._object_name,
        "success": hold_success,
        "time_to_grasp_s": (first_holding_step * dt) if reached_holding else None,
        "reached_holding": reached_holding,
        "had_crush": had_crush,
        "n_drop_events": len(drop_events),
        "n_drops_recovered": sum(drop_events),
        "grip_forces_while_holding": grip_forces_while_holding,
        "episode_length_steps": step_idx,
    }


# ----------------------------------------------------------------------
# Aggregation: turn a list of per-episode records into the 8 metrics
# ----------------------------------------------------------------------
def compute_metrics(records):
    """
    records: list of dicts from collect_episode_record(), assumed to all be
    for the SAME object (call this once per object, and again on the
    concatenation of all test-object records for the generalization score).

    Returns a dict with metrics 1, 2, 4, 5, 6, 8 (everything computable from
    rollouts alone -- metric 3 needs a training log, see
    compute_convergence_speed(); metric 7 is just this function called on
    the pooled test-object records, done in main()).
    """
    n = len(records)
    if n == 0:
        return {}

    n_success = sum(r["success"] for r in records)
    grasp_success_rate = 100.0 * n_success / n

    grasp_times = [r["time_to_grasp_s"] for r in records if r["success"] and r["time_to_grasp_s"] is not None]
    mean_time_to_grasp = float(np.mean(grasp_times)) if grasp_times else float("nan")

    n_failed = n - n_success
    failed_grasp_count_per_20 = (n_failed / n) * 20.0

    reached_holding_records = [r for r in records if r["reached_holding"]]
    n_reached_holding = len(reached_holding_records)
    n_with_a_drop = sum(1 for r in reached_holding_records if r["n_drop_events"] > 0)
    object_drop_rate = (100.0 * n_with_a_drop / n_reached_holding) if n_reached_holding > 0 else float("nan")

    all_holding_forces = [f for r in records for f in r["grip_forces_while_holding"]]
    avg_grip_force = float(np.mean(all_holding_forces)) if all_holding_forces else float("nan")

    total_drops = sum(r["n_drop_events"] for r in records)
    total_recovered = sum(r["n_drops_recovered"] for r in records)
    recovery_rate = (100.0 * total_recovered / total_drops) if total_drops > 0 else float("nan")

    n_crushed = sum(r["had_crush"] for r in records)

    return {
        "n_episodes": n,
        "grasp_success_rate_pct": round(grasp_success_rate, 2),
        "mean_time_to_grasp_s": round(mean_time_to_grasp, 3) if not np.isnan(mean_time_to_grasp) else None,
        "failed_grasp_count_per_20": round(failed_grasp_count_per_20, 2),
        "object_drop_rate_pct": round(object_drop_rate, 2) if not np.isnan(object_drop_rate) else None,
        "avg_grip_force_n": round(avg_grip_force, 3) if not np.isnan(avg_grip_force) else None,
        "recovery_rate_pct": round(recovery_rate, 2) if not np.isnan(recovery_rate) else None,
        "n_crushed": n_crushed,
    }


# ----------------------------------------------------------------------
# Metric 3: RL convergence speed, from a training run's monitor.csv
# ----------------------------------------------------------------------
def compute_convergence_speed(monitor_csv_path, window=100, threshold=0.8):
    """
    "Episodes required to reach 80% success on training objects" (Section
    5.4, metric: RL convergence speed).

    Reads an SB3 Monitor log written with info_keywords=("success",) (see
    train.py's make_env()). SB3's Monitor format: first line is a '#'
    comment with JSON metadata, second line is the CSV header, remaining
    lines are one row per completed episode.

    Returns the episode index (1-indexed) of the first episode whose
    trailing `window`-episode rolling average of `success` first reaches
    `threshold`, or None if that was never reached in this log.
    """
    if not os.path.exists(monitor_csv_path):
        return None

    successes = []
    with open(monitor_csv_path, "r", newline="") as f:
        first_line = f.readline()
        if not first_line.startswith("#"):
            f.seek(0)  # not an SB3 Monitor file after all; try parsing as plain CSV
        reader = csv.DictReader(f)
        for row in reader:
            if "success" not in row:
                raise ValueError(
                    f"{monitor_csv_path} has no 'success' column -- was it written "
                    "with Monitor(env, ..., info_keywords=('success',))? "
                    "See train.py's make_env()."
                )
            successes.append(row["success"].strip().lower() in ("true", "1"))

    if len(successes) < window:
        return None

    for i in range(window, len(successes) + 1):
        rolling_mean = sum(successes[i - window:i]) / window
        if rolling_mean >= threshold:
            return i

    return None


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_evaluation(action_fn, gripper_type, mode, stage, object_set,
                    episodes_per_object, episode_length, dt, seed,
                    hold_threshold_seconds, recovery_window_seconds,
                    action_repeat=1):
    """
    Runs `episodes_per_object` episodes for every object in `object_set`
    ("train", "test", or "both"), one fresh NazManipulationEnv per object
    (so different objects can't leak state between each other), and
    returns (per_object_records, per_object_metrics, pooled_metrics_by_set).
    """
    if object_set == "train":
        object_names = list(TRAINING_OBJECTS.keys())
    elif object_set == "test":
        object_names = list(TEST_OBJECTS.keys())
    elif object_set == "both":
        object_names = list(TRAINING_OBJECTS.keys()) + list(TEST_OBJECTS.keys())
    else:
        raise ValueError(f"object_set must be 'train', 'test', or 'both', got {object_set!r}")

    per_object_records = {}
    for obj_name in object_names:
        env = NazManipulationEnv(
            gripper_type=gripper_type, mode=mode, stage=stage,
            episode_length=episode_length, dt=dt, object_names=[obj_name],
        )
        records = [
            collect_episode_record(
                env, action_fn, dt,
                hold_threshold_seconds=hold_threshold_seconds,
                recovery_window_seconds=recovery_window_seconds,
                seed=seed + ep,
                action_repeat=action_repeat,
            )
            for ep in range(episodes_per_object)
        ]
        per_object_records[obj_name] = records

    per_object_metrics = {name: compute_metrics(recs) for name, recs in per_object_records.items()}

    pooled_train_records = [r for name in TRAINING_OBJECTS if name in per_object_records
                             for r in per_object_records[name]]
    pooled_test_records = [r for name in TEST_OBJECTS if name in per_object_records
                            for r in per_object_records[name]]

    pooled = {}
    if pooled_train_records:
        pooled["OVERALL_TRAIN"] = compute_metrics(pooled_train_records)
    if pooled_test_records:
        # This pooled test-set success rate IS metric 7, "generalization score" --
        # named explicitly here rather than left implicit, since the brief calls
        # it out as the primary generalization metric.
        test_metrics = compute_metrics(pooled_test_records)
        test_metrics["generalization_score_pct"] = test_metrics["grasp_success_rate_pct"]
        pooled["OVERALL_TEST_GENERALIZATION"] = test_metrics

    return per_object_records, per_object_metrics, pooled


def write_csv(out_path, per_object_metrics, pooled_metrics, convergence_episodes):
    fieldnames = ["object", "n_episodes", "grasp_success_rate_pct", "mean_time_to_grasp_s",
                  "failed_grasp_count_per_20", "object_drop_rate_pct", "avg_grip_force_n",
                  "recovery_rate_pct", "n_crushed", "rl_convergence_episodes"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in list(per_object_metrics.items()) + list(pooled_metrics.items()):
            row = {"object": name, "rl_convergence_episodes": convergence_episodes}
            row.update({k: v for k, v in metrics.items() if k in fieldnames})
            writer.writerow(row)
    print(f"[eval.py] Wrote {out_path}")


# ----------------------------------------------------------------------
# SB3-dependent entry point
# ----------------------------------------------------------------------
def load_model(model_path, algo):
    try:
        from stable_baselines3 import SAC, PPO
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required to load a trained model. Install with:\n"
            '    pip install "stable-baselines3[extra]" gymnasium'
        ) from e
    algo_cls = {"SAC": SAC, "PPO": PPO}[algo]
    return algo_cls.load(model_path)


def main(argv=None):
    from train import CONDITION_MAP  # reuse the single source of truth for A/B/C/D

    p = argparse.ArgumentParser(description="Evaluate a trained Project Naz policy.")
    p.add_argument("--condition", required=True, choices=list(CONDITION_MAP.keys()))
    p.add_argument("--stage", type=int, default=1, choices=range(1, 7), metavar="1-6")
    p.add_argument("--model-path", required=True)
    p.add_argument("--algo", choices=["SAC", "PPO"], default="SAC")
    p.add_argument("--monitor-csv", default=None,
                    help="Path to the training run's monitor.csv, for the RL convergence speed metric.")
    p.add_argument("--object-set", choices=["train", "test", "both"], default="both")
    p.add_argument("--episodes-per-object", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--hold-threshold-seconds", type=float, default=3.0)
    p.add_argument("--recovery-window-seconds", type=float, default=1.5)
    p.add_argument("--action-repeat", type=int, default=1,
                    help="Must match whatever --action-repeat the model was trained with "
                         "(see train.py). Re-invokes the policy only every N raw env steps.")
    p.add_argument("--out", default="eval_results.csv")
    args = p.parse_args(argv)

    gripper_type, mode = CONDITION_MAP[args.condition]
    model = load_model(args.model_path, args.algo)
    action_fn = lambda obs: model.predict(obs, deterministic=True)[0]

    _, per_object_metrics, pooled = run_evaluation(
        action_fn, gripper_type, mode, args.stage, args.object_set,
        args.episodes_per_object, args.episode_length, args.dt, args.seed,
        args.hold_threshold_seconds, args.recovery_window_seconds,
        action_repeat=args.action_repeat,
    )

    convergence_episodes = None
    if args.monitor_csv:
        convergence_episodes = compute_convergence_speed(args.monitor_csv)
        print(f"[eval.py] RL convergence speed (episodes to 80% success): {convergence_episodes}")

    write_csv(args.out, per_object_metrics, pooled, convergence_episodes)

    for name, m in pooled.items():
        print(f"\n{name}: {m}")


if __name__ == "__main__":
    main()