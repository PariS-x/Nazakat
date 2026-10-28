"""
check_stage3_hold.py -- direct diagnostic for Stage 3 (Grip Discovery).

Unlike check_reach.py (which checks the loose 6cm "contact radius" from
Stage 2), Stage 3's own reward is silently gated on actually reaching
GripStatus.HOLDING at least once -- since object displacement in this
simulation only happens while the gripper is genuinely holding the object,
not merely touching it. This measures that directly: reached_holding rate,
deterministic policy, over a real sample size (default 100 episodes) rather
than the 5 episodes the live training callback uses.

USAGE:
    python check_stage3_hold.py <model_path> --stage 3 --gripper rigid

EXAMPLE:
    python check_stage3_hold.py runs/C_stage3_seed0/checkpoints/naz_900000_steps.zip --stage 3 --gripper rigid
"""
import argparse
import numpy as np
from stable_baselines3 import SAC
from naz_env import NazManipulationEnv
from eval import collect_episode_record

p = argparse.ArgumentParser()
p.add_argument("model_path")
p.add_argument("--stage", type=int, default=3)
p.add_argument("--gripper", choices=["rigid", "soft"], default="rigid")
p.add_argument("--action-repeat", type=int, default=50)
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--episodes", type=int, default=100)
args = p.parse_args()

model = SAC.load(args.model_path)
env = NazManipulationEnv(gripper_type=args.gripper, mode="developmental", stage=args.stage,
                          object_names=None, episode_length=500)
action_fn = lambda obs: model.predict(obs, deterministic=True)[0]

reached_holding = 0
sustained_success = 0
crushed = 0
never_touched = 0

for ep in range(args.episodes):
    rec = collect_episode_record(env, action_fn, dt=args.dt, seed=ep, action_repeat=args.action_repeat)
    if rec["reached_holding"]:
        reached_holding += 1
    if rec["success"]:
        sustained_success += 1
    if rec["had_crush"]:
        crushed += 1
    if not rec["reached_holding"] and not rec["had_crush"]:
        never_touched += 1

print(f"Model: {args.model_path}  (stage {args.stage}, {args.gripper} gripper)")
print(f"Episodes: {args.episodes}")
print(f"Reached HOLDING at least once: {reached_holding}/{args.episodes}")
print(f"Sustained success (>=3s hold, Stage 4-6 bar, not expected yet): {sustained_success}/{args.episodes}")
print(f"Crushed: {crushed}/{args.episodes}")
print(f"Never even touched/held: {never_touched}/{args.episodes}")
