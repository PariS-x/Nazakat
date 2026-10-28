# eval_A0.py
from stable_baselines3 import SAC
from eval import run_evaluation, write_csv, compute_convergence_speed

RUN_DIR = "runs/A_std_seed0"

# Must match whatever --action-repeat train.py used for this run (default 50
# as of the exploration-dead-end fix -- see env_wrappers.py). If you trained
# this run before that fix (action_repeat=1 / no wrapper), set this to 1.
ACTION_REPEAT = 50

model = SAC.load(f"{RUN_DIR}/final_model.zip")

def action_fn(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

per_object_records, per_object_metrics, pooled = run_evaluation(
    action_fn,
    gripper_type="rigid",
    mode="standard",
    stage=1,
    object_set="both",
    episodes_per_object=20,
    episode_length=500,
    dt=0.01,
    seed=0,
    hold_threshold_seconds=3.0,
    recovery_window_seconds=1.5,
    action_repeat=ACTION_REPEAT,
)

print("TRAIN objects:", pooled["OVERALL_TRAIN"])
print("TEST objects (generalization):", pooled["OVERALL_TEST_GENERALIZATION"])

# write_csv needs the training run's convergence-speed number too (this was
# the actual crash: the call below was missing this 4th argument entirely).
convergence_episodes = compute_convergence_speed(f"{RUN_DIR}/monitor.csv")
print("RL convergence speed (episodes to 80% success):", convergence_episodes)

write_csv(f"{RUN_DIR}/eval_results.csv", per_object_metrics, pooled, convergence_episodes)
print(f"Full breakdown written to {RUN_DIR}/eval_results.csv")