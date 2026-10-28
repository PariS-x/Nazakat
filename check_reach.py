import argparse
import numpy as np
from stable_baselines3 import SAC
from naz_env import NazManipulationEnv

CONTACT_RADIUS_M = 0.06  # matches naz_env.py's own CONTACT_RADIUS_M

p = argparse.ArgumentParser()
p.add_argument("model_path")
p.add_argument("--stage", type=int, default=2)
p.add_argument("--gripper", choices=["rigid", "soft"], default="rigid")
p.add_argument("--action-repeat", type=int, default=50)
p.add_argument("--episodes", type=int, default=100)
args = p.parse_args()

model = SAC.load(args.model_path)
env = NazManipulationEnv(gripper_type=args.gripper, mode="developmental", stage=args.stage,
                          object_names=None, episode_length=500)

closest_per_ep = []
contact_events = 0
for ep in range(args.episodes):
    obs, _ = env.reset(seed=ep)
    min_dist = np.inf
    action = None
    for t in range(500):
        if t % args.action_repeat == 0:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        d = np.linalg.norm(info["object_position"] - env.platform.ee_world_position())
        min_dist = min(min_dist, d)
        if term or trunc:
            break
    closest_per_ep.append(min_dist)
    if min_dist <= CONTACT_RADIUS_M:
        contact_events += 1

closest_per_ep = np.array(closest_per_ep)
print(f"Model: {args.model_path}  (stage {args.stage}, {args.gripper} gripper)")
print(f"Episodes: {args.episodes}")
print(f"Closest approach -- mean: {closest_per_ep.mean():.4f}m  median: {np.median(closest_per_ep):.4f}m  "
      f"min: {closest_per_ep.min():.4f}m  max: {closest_per_ep.max():.4f}m")
print(f"Episodes reaching within {CONTACT_RADIUS_M}m contact radius: {contact_events}/{args.episodes}")