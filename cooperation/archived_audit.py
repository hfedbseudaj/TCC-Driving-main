"""Read-only archived-model E0 audit on the four original two-car scenes.

The archive source is compiled in memory and never imported through a loader
that writes bytecode beside it.  The archived test_bargaining.py simulator and
this project's copy differ only in JSON serialization, so the shared SimState
can execute the archived model without changing its physical update rule.
Three/four-car author scripts and trained weights are not supplied by this
entry point; their absence is recorded instead of inferred from these trials.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import types
import uuid

import numpy as np
import torch

from test_bargaining import SCENARIOS, SimState


_ROOT = Path(__file__).resolve().parent.parent


def _archive_module(path):
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError("Archived bargaining source not found: {}".format(source))
    module = types.ModuleType("read_only_archived_rubinstein_bargaining")
    module.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def _default_archive_source():
    for candidate in (
            _ROOT.parent / "TCC-Driving-main1" / "TCC-Driving-main",
            _ROOT.parent / "TCC-Driving-main1"):
        path = candidate / "opencood" / "models" / "comm_modules" / "rubinstein_bargaining.py"
        if path.is_file():
            return path
    return None


def _weights(model, checkpoint_path):
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path).resolve()
    blob = torch.load(str(path), map_location="cpu")
    metadata = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_seed": blob.get("training_seed", blob.get("seed")) if isinstance(blob, dict) else None,
        "checkpoint_schema_version": blob.get("checkpoint_schema_version") if isinstance(blob, dict) else None,
        "model_config": blob.get("model_config") if isinstance(blob, dict) else None,
        "observation_spec": blob.get("observation_spec") if isinstance(blob, dict) else None,
    }
    if isinstance(blob, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break
    if not isinstance(blob, dict):
        raise ValueError("Checkpoint has no state dictionary")
    model.load_state_dict(blob, strict=True)
    return metadata


def _safe_json(value):
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity"
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_safe_json(value), ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _project_legacy_action(sim, state, adjusted_waypoints):
    """Reproduce the archived test script's second-car fallback exactly."""
    base = state["waypoints"].detach().cpu()
    adjusted = adjusted_waypoints.detach().cpu()
    applied = []
    for vehicle in range(2):
        if vehicle < adjusted.shape[0]:
            delta = adjusted[vehicle, 0] - base[vehicle, 0]
        else:
            delta = -(adjusted[0, 0] - base[0, 0]) * 0.5
        xy = [float(delta[0]), float(delta[1])]
        sim.adjust(vehicle, xy[0], xy[1])
        applied.append(xy)
    return applied


def audit_trial(module, scenario, model_seed, trial_seed, max_steps,
                device, checkpoint_path=None, position_std=0.0,
                speed_std=0.0, heading_std=0.0,
                feature_mode="original_random_bev"):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown archived two-car scenario: {}".format(scenario))
    torch.manual_seed(int(model_seed))
    model = module.RubinsteinBargaining({
        "state_dim": 64, "offer_dim": 4, "hidden_dim": 128,
        "max_rounds": 3, "discount_alpha": 0.3,
        "delta_min": 0.05, "delta_max": 0.99,
        "waypoint_offset_scale": 2.0,
    })
    checkpoint_info = _weights(model, checkpoint_path)
    model = model.to(device).eval()
    sim = SimState(scenario, dt=0.1)
    perturbations = None
    if any(value != 0.0 for value in (position_std, speed_std, heading_std)):
        from common.experiment_protocol import perturb_scenario
        perturbations = perturb_scenario(
            sim, trial_seed, position_std, speed_std, heading_std)
    features_rng = torch.Generator(device=device)
    features_rng.manual_seed(int(trial_seed))
    records = []
    for step in range(int(max_steps)):
        if sim.finished:
            break
        state = sim.get_state()
        positions = state["positions"].to(device)
        velocities = state["velocities"].to(device)
        headings = state["headings"].to(device)
        waypoints = state["waypoints"].to(device)
        record_len = state["record_len"].to(device)
        if feature_mode == "original_random_bev":
            features = torch.randn((2, 256, 48, 48), device=device,
                                   generator=features_rng)
        elif feature_mode == "deterministic_pair_bev":
            from common.deterministic_bev import deterministic_bev
            features = deterministic_bev(state, device=device)
        else:
            raise ValueError("Unsupported feature mode")
        with torch.no_grad():
            encoded, _ = model.encode_states(features)
            raw_offers = model.offer_decoder(encoded)
            ttc = module.compute_pairwise_ttc(positions, velocities, headings)
            discounts = model.discount(ttc)
            output = model(features, waypoints, positions, velocities,
                           headings, record_len)
        applied = _project_legacy_action(sim, state, output["adjusted_waypoints"])
        before = [item.tolist() for item in sim.pos]
        sim.step_sim()
        distance_m = float(np.linalg.norm(sim.pos[0] - sim.pos[1]))
        records.append({
            "step": step,
            "positions_before_m": before,
            "positions_after_m": [item.tolist() for item in sim.pos],
            "speeds_after_mps": [float(value) for value in sim.spd],
            "ttc_s": float(ttc[0, 1].item()),
            "discount_factors": [float(value) for value in discounts.min(dim=1)[0].tolist()],
            "raw_offers": raw_offers.detach().cpu().tolist(),
            "negotiated_offers": output["bargaining_offers"].detach().cpu().tolist(),
            "applied_first_waypoint_offsets_m": applied,
            "rounds": int(output["metadata"][0]["n_rounds"]),
            "converged": bool(output["metadata"][0]["converged"]),
            "distance_after_m": distance_m,
            "collision_after_step": bool(sim.collision),
        })
    min_distance = min((row["distance_after_m"] for row in records),
                       default=math.inf)
    return {
        "scenario": scenario,
        "model_seed": int(model_seed),
        "trial_seed": int(trial_seed),
        "training_seed": checkpoint_info["training_seed"] if checkpoint_info else None,
        "checkpoint_info": checkpoint_info,
        "checkpoint_sha256": checkpoint_info["sha256"] if checkpoint_info else None,
        "model_status": "untrained_random_initialization" if checkpoint_path is None else "checkpoint_loaded_strictly",
        "feature_mode": feature_mode,
        "perturbations": perturbations,
        "steps": len(records),
        "collision": bool(sim.collision),
        "near_collision": bool(min_distance < 2.2 and not sim.collision),
        "min_sampled_distance_m": min_distance,
        "records": records,
    }


def run_audit(archive_source, output_root, model_seeds, trial_seeds,
              max_steps, device="cpu", checkpoint_path=None,
              feature_mode="original_random_bev", position_std=0.0,
              speed_std=0.0, heading_std=0.0):
    archive_source = Path(archive_source).resolve()
    output_root = Path(output_root).resolve()
    if _ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    if not model_seeds or not trial_seeds or max_steps < 1:
        raise ValueError("Provide model seeds, trial seeds, and positive max_steps")
    module = _archive_module(archive_source)
    started = time.perf_counter()
    rows = []
    for scenario in SCENARIOS:
        for model_seed in model_seeds:
            for trial_seed in trial_seeds:
                rows.append(audit_trial(
                    module, scenario, model_seed, trial_seed, max_steps,
                    device, checkpoint_path, position_std, speed_std,
                    heading_std, feature_mode))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    aggregate = defaultdict(list)
    for row in rows:
        aggregate[row["scenario"]].append(row)
    summary = {
        scenario: {
            "trials": len(group),
            "collisions": sum(int(row["collision"]) for row in group),
            "near_collisions": sum(int(row["near_collision"]) for row in group),
            "collision_rate": sum(int(row["collision"]) for row in group) / len(group),
            "mean_min_sampled_distance_m": sum(row["min_sampled_distance_m"] for row in group) / len(group),
        }
        for scenario, group in aggregate.items()
    }
    _write_json(output_dir / "trials.json", rows)
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "audit_target": "read_only_archived_original_module_four_two_car_scenes",
        "archive_source": str(archive_source),
        "archive_source_sha256": hashlib.sha256(archive_source.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulator_source_sha256": hashlib.sha256((_ROOT / "test_bargaining.py").read_bytes()).hexdigest(),
        "archive_modified": False,
        "author_multicar_script_available": False,
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "checkpoint_info": rows[0]["checkpoint_info"],
        "checkpoint_status": rows[0]["model_status"],
        "model_initialization_seeds": list(model_seeds),
        "scenario_trial_seeds": list(trial_seeds),
        "trial_count": len(rows),
        "feature_mode": feature_mode,
        "scenario_perturbation_std": {
            "position_m": position_std, "speed_mps": speed_std,
            "heading_rad": heading_std,
        },
        "max_steps": max_steps,
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "elapsed_s": time.perf_counter() - started,
        "scope_note": "Original code has no per-edge multi-car contracts; these two-car trials cannot diagnose a cause of the paper's 3/4-car collisions.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-source", type=Path,
                        default=_default_archive_source())
    parser.add_argument("--output-root", type=Path,
                        default=_ROOT / "Result" / "archived_original_audit")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--feature-mode", choices=("original_random_bev", "deterministic_pair_bev"),
                        default="original_random_bev")
    parser.add_argument("--position-std", type=float, default=0.0)
    parser.add_argument("--speed-std", type=float, default=0.0)
    parser.add_argument("--heading-std", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.archive_source is None:
        parser.error("Archived source not found; pass --archive-source")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this virtual environment")
    output_dir, summary = run_audit(
        args.archive_source, args.output_root, args.model_seeds,
        args.trial_seeds, args.max_steps, args.device, args.checkpoint,
        args.feature_mode, args.position_std, args.speed_std,
        args.heading_std)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
