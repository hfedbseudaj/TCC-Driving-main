"""Shared reproducibility helpers for SL validation and Table 4 evaluation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def derive_trial_seed(base_seed, scenario, trial_index, scenario_names):
    """Return a stable scene seed shared by every model in a comparison."""
    scenario_index = list(scenario_names).index(scenario)
    return int(base_seed) + scenario_index * 100_000 + int(trial_index)


def perturb_scenario(sim, seed, position_std=0.75, speed_std=0.6, heading_std=0.035):
    """Apply a reproducible, bounded scale of initial state noise in-place."""
    rng = np.random.RandomState(int(seed))
    changes = {"position_xy": [], "speed": [], "heading_rad": []}
    for vehicle in range(2):
        xy_noise = rng.normal(0.0, position_std, size=2).astype(np.float32)
        speed_noise = float(rng.normal(0.0, speed_std))
        heading_noise = float(rng.normal(0.0, heading_std))
        sim.pos[vehicle] = sim.pos[vehicle] + xy_noise
        sim.spd[vehicle] = max(0.3, float(sim.spd[vehicle]) + speed_noise)
        sim.hdg[vehicle] = float(sim.hdg[vehicle]) + heading_noise
        changes["position_xy"].append([float(x) for x in xy_noise])
        changes["speed"].append(speed_noise)
        changes["heading_rad"].append(heading_noise)
    sim._update_wps()
    return changes


def selection_score(rows, latency_budget_ms=50.0):
    """Lower is better; retain every safety, convergence and latency term."""
    if not rows:
        raise ValueError("Cannot score an empty validation set")
    n = len(rows)
    collision_rate = sum(bool(r["collision"]) for r in rows) / n
    near_rate = sum(bool(r["near_collision"]) for r in rows) / n
    ttc_risk = sum(float(r["ttc_risk"]) for r in rows) / n
    convergence_rate = sum(float(r["conv"]) for r in rows) / n / 100.0
    latency_ms = sum(float(r["latency_ms"]) for r in rows) / n
    latency_norm = min(max(latency_ms / max(latency_budget_ms, 1e-6), 0.0), 1.0)
    score = (100.0 * collision_rate + 25.0 * near_rate + 20.0 * ttc_risk
             + 10.0 * (1.0 - convergence_rate) + 5.0 * latency_norm)
    return {
        "selection_score": float(score),
        "collision_rate": float(collision_rate),
        "near_collision_rate": float(near_rate),
        "mean_ttc_risk": float(ttc_risk),
        "convergence_rate": float(convergence_rate),
        "mean_latency_ms": float(latency_ms),
        "validation_episodes": n,
        "latency_budget_ms": float(latency_budget_ms),
        "weights": {
            "collision_rate": 100.0,
            "near_collision_rate": 25.0,
            "ttc_risk": 20.0,
            "convergence_failure": 10.0,
            "latency_fraction_of_budget": 5.0,
        },
    }


def project_fingerprint(project_root, source_files):
    """Return a git revision when available and a content digest in all cases."""
    root = Path(project_root)
    digest = hashlib.sha256()
    found = []
    for relative in sorted(source_files):
        path = root / relative
        if path.is_file():
            found.append(relative)
            digest.update(relative.replace("\\", "/").encode("utf-8"))
            digest.update(path.read_bytes())
    revision = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
            universal_newlines=True, timeout=5, check=False)
        if result.returncode == 0:
            revision = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": revision, "source_sha256": digest.hexdigest(),
            "fingerprinted_files": found}


def environment_metadata(torch_module, numpy_module=np):
    cuda_available = bool(torch_module.cuda.is_available())
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch_module.cuda.get_device_name(0)
        except Exception:
            gpu_name = "available"
    return {
        "python": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "torch": str(torch_module.__version__),
        "numpy": str(numpy_module.__version__),
        "cuda_available": cuda_available,
        "torch_cuda": getattr(torch_module.version, "cuda", None),
        "cudnn": (torch_module.backends.cudnn.version()
                  if cuda_available and torch_module.backends.cudnn.is_available() else None),
        "gpu": gpu_name,
    }


def episode_metrics(collision, near_collision, min_distance, min_ttc, conv_rate,
                    latency_ms, steps, ttc_threshold=3.0):
    """Normalize episode measurements for reporting and selection."""
    min_ttc = float(min_ttc)
    ttc_risk = min(max((ttc_threshold - min_ttc) / ttc_threshold, 0.0), 1.0)
    return {
        "collision": bool(collision),
        "near_collision": bool(near_collision),
        "min_distance": float(min_distance),
        "min_ttc": min_ttc,
        "ttc_risk": float(ttc_risk),
        "conv": float(conv_rate),
        "latency_ms": float(latency_ms),
        "steps": int(steps),
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
