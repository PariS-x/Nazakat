"""
Diagnostic: is the Stage 2 policy actually homing toward the object,
or is it moving around without regard to where the object is?

Run this on your machine (needs stable-baselines3 + your project files):
    python check_stage2_reach.py runs/C_stage2_seed0/final_model.zip

Measures closest end-effector-to-object approach per episode, deterministic
policy, same action_repeat as training -- directly answers "does it get
close" instead of inferring it from monitor.csv reward totals.
"""
import sys
import numpy as np
from stable_baselines3 import SAC
from naz_env import NazManipulationEnv

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/C_stage2_seed0/final_model.zip"
ACTION_REPEAT = 50
N_EPISODES = 100

model = SAC.load(model_path)
env = NazManipulationEnv(gripper_type="rigid", mode="developmental", stage=2,
                          object_names=None, episode_length=500)

closest_per_ep = []
contact_events = 0
for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=ep)
    min_dist = np.inf
    action = None
    for t in range(500):
        if t % ACTION_REPEAT == 0:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        d = np.linalg.norm(info["object_position"] - env.platform.ee_world_position())
        min_dist = min(min_dist, d)
        if term or trunc:
            break
    closest_per_ep.append(min_dist)
    if min_dist <= 0.06:
        contact_events += 1

closest_per_ep = np.array(closest_per_ep)
print(f"Episodes: {N_EPISODES}")
print(f"Closest approach -- mean: {closest_per_ep.mean():.4f}m  median: {np.median(closest_per_ep):.4f}m  "
      f"min: {closest_per_ep.min():.4f}m  max: {closest_per_ep.max():.4f}m")
print(f"Episodes reaching within 6cm contact radius: {contact_events}/{N_EPISODES}")
