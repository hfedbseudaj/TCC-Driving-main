# AdvancedCoLM

> TTC-calibrated differentiable Rubinstein bargaining for cooperative autonomous driving.

## Overview

AdvancedCoLM adds an explicit bargaining layer to a cooperative driving pipeline based on CoLMDriver. The core module maps physical urgency, represented by time-to-collision (TTC), to a bargaining discount factor:

```text
delta = 1 - exp(-alpha * TTC)
```

The bargaining layer predicts and evaluates trajectory offers, executes a differentiable alternating-offers protocol, and converts the agreed offer into local waypoint or motion adjustments.

This repository contains the core pairwise bargaining implementation, training and ablation scripts, cooperative perception/planning code, and the CARLA integration path. It also contains an opt-in, deterministic crossing-scene prototype for the new multi-vehicle cooperation research plan. That prototype is separate from the trained bargaining and CARLA execution paths.

## Quick Start

Create the environment, install the dependencies, and run the dependency-free
kinematic smoke test from the repository root:

```powershell
conda create -n advancedcolm python=3.7
conda activate advancedcolm
python -m pip install -r requirements.txt
python -m pip install -r opencood/requirements.txt
python -m pip install -r simulation/requirements.txt
python test_bargaining.py --all-variants --no-plot
```

The smoke test writes generated logs and figures under `Result/`, which is
ignored by Git. It does not require CARLA or pretrained checkpoints.

## Repository Structure

```text
.
├── common/                 Shared utilities
├── cooperation/            Opt-in multi-vehicle crossing-contract prototype
├── config/                 Constructed cooperation example configuration
├── codriving/              Co-driving models, data utilities, and losses
├── opencood/               Cooperative perception and planning framework
│   └── models/comm_modules/
│       └── rubinstein_bargaining.py
├── simulation/             CARLA / ScenarioRunner integration
├── vision_encoder/         Vision encoder components
├── MLLMs/                  Offline language-model data utilities
├── scripts/                Training and evaluation shell scripts
├── visualization/          Visualization helpers
├── img/                    Documentation figures
├── train_bargaining_supervised.py
├── train_bargaining_ppo.py
├── train_bargaining_rl.py
├── test_bargaining.py
├── test_bargaining_ablation.py
├── setup.py
└── requirements.txt
```

## Main Components

### Differentiable bargaining

The core module provides:

- Pairwise TTC computation.
- TTC-based discount factors.
- Neural utility estimation.
- Offer decoding.
- Differentiable alternating offers.
- Waypoint and motion-adjustment decoding.

Main files:

```text
opencood/models/comm_modules/rubinstein_bargaining.py
opencood/models/center_point_codriving_with_bargaining.py
codriving/losses/bargaining.py
```

### Training

The release includes supervised learning, PPO, and self-play reinforcement learning routes:

```powershell
python train_bargaining_supervised.py --epochs 30
python train_bargaining_ppo.py --epochs 200
python train_bargaining_rl.py --epochs 200
```

The supervised route uses a rule-based expert to stabilize the bargaining module. PPO and self-play are provided for closed-loop policy exploration.

### Evaluation and ablation

```powershell
python test_bargaining.py --all-variants
python test_bargaining_ablation.py
python test_bargaining_ablation_trained.py --compare
```

Typical metrics include collision rate, convergence rate, utility consistency, minimum distance, delay, intervention magnitude, and offer loss.

### Multi-vehicle cooperation prototype

The new research-plan implementation starts from the paper's two-car TTC discount and closed-form Rubinstein surplus share. For the crossing scene, it decodes that share into a reachable pair trajectory contract, then checks whether contracts on multiple conflict edges can be jointly satisfied. If they cannot, it selects a plan by a documented finite-candidate repair order. The package now also contains a read-only archived-model audit, an explicit old-offer adapter, paired kinematic closed-loop trials, and provisional scene-specific definitions for following, merging, and head-on approach.

```powershell
python -m cooperation.run_snapshot --allow-constructed-example
python -m cooperation.archived_audit --max-steps 80
python -m cooperation.run_paired_closed_loop --allow-constructed-example
python -m cooperation.run_adapter_audit --allow-constructed-example
python -m cooperation.run_scene_extensions
python -m unittest tests.test_multicar_cooperation
```

The bundled configuration contains **constructed teaching numbers**, including one manually specified second-edge contract. Snapshot, paired-loop, adapter, and scene-extension results under `Result/` must be read with their manifests: they do not establish trained-policy superiority, CARLA safety, or reproduction of the paper's multi-car collision rate. See [cooperation/README.md](cooperation/README.md) for definitions and the [2026-11-20 experiment report](cooperation/EXPERIMENT_REPORT_2026-11-20.md) for completed runs, failures, and remaining work.

## Installation

The original CoLMDriver stack uses a Python 3.7 environment. Select CUDA and PyTorch versions according to the target machine and CARLA installation.

```powershell
conda create -n advancedcolm python=3.7
conda activate advancedcolm
python -m pip install -r requirements.txt
python -m pip install -r opencood/requirements.txt
python -m pip install -r simulation/requirements.txt
python -m pip install -e .
```

Some OpenCOOD modules require compiled CUDA extensions:

```powershell
python opencood/utils/setup.py build_ext --inplace
```

CARLA is not bundled in this repository. Install a compatible CARLA release separately and configure the external CARLA path before running CARLA-based evaluation.

The public repository also does not include perception/planning checkpoints.
For closed-loop evaluation, place compatible weights under the ignored
`ckpt/` directory or pass the corresponding paths to the evaluation launcher.
Set `CARLA_ROOT` to the external CARLA installation when the launcher needs the
Python API:

```powershell
$env:CARLA_ROOT = "C:\path\to\CARLA"
python server_preflight.py --skip-carla
python launch_closed_loop_suite.py --variant suite --carla-port 2000 --dry-run
```

## Reproducibility and Scope

The public release intentionally excludes:

- Complete N-player coalition and hypergraph aggregation experiments.
- `Result/` outputs, checkpoints, local caches, and generated frames.
- CARLA binary distributions.
- Python bytecode, logs, temporary files, and compiled artifacts.

The release should therefore be interpreted as a compact core implementation of TTC-calibrated pairwise bargaining integrated with cooperative driving, plus a separate crossing-scene research prototype. It does not claim to solve a complete N-player dynamic game.

The CARLA code still contains the original cooperative driving data and
multi-vehicle execution interfaces inherited from CoLMDriver. These are
baseline pipeline interfaces, not the excluded N-player coalition bargaining
research branch.

## Citation

Please cite the original CoLMDriver work and the associated bargaining research when using this repository:

```bibtex
@inproceedings{hu2023colmdriver,
  title     = {CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving},
  booktitle = {International Conference on Computer Vision},
  year      = {2025}
}
```

## License

See `LICENSE` and the licenses of the upstream components included in this repository.

