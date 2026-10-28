# Nazakat: Compliant Morphology and Developmental Learning for Robotic Manipulation

A controlled **2 × 2 factorial simulation study** of what makes robotic grasping of unknown objects robust:
gripper morphology (rigid vs. cephalopod-inspired soft) crossed with learning strategy
(sparse-reward RL vs. a six-stage infant-inspired developmental curriculum), trained with Soft Actor-Critic
and evaluated on 12 objects over 3 random seeds per condition.

## Results (generalisation to 9 unseen objects, mean ± std over 3 seeds)

| Condition | Gripper | Training | Train success | Unseen success |
|---|---|---|---|---|
| A | Rigid | Standard RL | 8.3 ± 10.4% | 4.6 ± 6.2% |
| B | Soft | Standard RL | 45.0 ± 47.7% | 36.5 ± 42.1% |
| C | Rigid | Developmental | 0.0 ± 0.0% | 0.0 ± 0.0% |
| D | Soft | Developmental | 21.7 ± 11.5% | 18.1 ± 10.5% |

The soft gripper beat the rigid one in every seed-matched comparison. The curriculum's early stages taught
reaching and grip discovery, but an uncapped force-overshoot penalty in Stages 4–5 taught the policy to avoid
gripping. Per-seed CSVs are in the `results_*.csv` files.

## What is in this repo

| Layer | Files |
|---|---|
| Rover physics (Mars gravity, MOLA terrain) | `rover_components.py`, `dynamic_rover.py`, `mars_loader.py`, `controller.py`, `simulation_engine.py`, `visualizer.py` |
| 4-DOF arm (DH kinematics, IK, PD control) | `arm_model.py` |
| Grippers (shared interface, rigid, soft 6-pad) | `gripper_base.py`, `rigid_gripper.py`, `soft_gripper.py` |
| Integration | `manipulation_platform.py` |
| RL environment and action repeat | `naz_env.py`, `env_wrappers.py` |
| Training and evaluation | `train.py`, `eval.py`, `eval_A0.py` |
| Tests and diagnostics | `test_*.py`, `check_*.py` |
| Results | `results_*.csv`, `make_results_chart.py` |

## Setup

```bash
python -m venv naz_env
naz_env\Scripts\activate          # Windows  (Linux/macOS: source naz_env/bin/activate)
pip install -r requirements.txt
```

Terrain data: download the NASA MOLA MEGDR tile `megt00n000hb.img` (+ `.lbl`) from the
PDS Geosciences Node and place both in `mars_terrain/` (the file is too large for GitHub).

## Run

```bash
# train Condition B (soft gripper, standard RL), seed 0
python train.py --condition B --timesteps 600000 --seed 0

# train one curriculum stage (Condition D, Stage 2); later stages resume from the previous stage
python train.py --condition D --stage 2 --timesteps 600000 --seed 0

# evaluate a checkpoint on all 12 objects (20 episodes each)
python eval.py --condition B --model-path runs/B_std_seed0/final_model.zip --action-repeat 50 --out results_B.csv

# unit tests
python -m pytest
```

Trained models for the reported checkpoints are attached to the GitHub Release (too large for the repo).

## Citation

P. Sharma, V. Belsare, R. Jaiswal, R. Dalvi, P. Khare, M. Tiwari, “Nazakat: Compliant Morphology and
Developmental Learning for Robotic Manipulation,” submitted to ICSSS 2026 (under review).
