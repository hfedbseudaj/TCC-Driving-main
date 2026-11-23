"""E5 per-scene finite-candidate definitions and 2/3/4-car scale checks."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import uuid

from .run_snapshot import _write_json
from .scene_extensions import EXTENSION_FIXED_MODEL, evaluate_scene


_ROOT = Path(__file__).resolve().parent.parent


def run_extension_sweep(config_path, output_root):
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported E5 config schema")
    if config.get("status") != "new_provisional_E5_definitions_not_original_paper_parameters":
        raise ValueError("E5 provenance label is required")
    for key, actual in EXTENSION_FIXED_MODEL.items():
        if float(config["physical_model"][key]) != actual:
            raise ValueError("Config {} disagrees with implemented fixed model".format(key))
    if _ROOT.as_posix().startswith("/data/hhr/"):
        designated = Path("/data/hhr").resolve()
        if designated != output_root and designated not in output_root.parents:
            raise ValueError("Server output must stay under /data/hhr")
    rows = []
    for scene in config["scenes"]:
        for n_cars in config["vehicle_counts"]:
            for trial_seed in config["trial_seeds"]:
                try:
                    result = evaluate_scene(
                        scene,
                        alpha_per_s=float(config["discount_alpha_per_s"]),
                        eta_m=float(config["near_optimal_surplus_eta_m"]),
                        welfare_epsilon_m=float(config["welfare_tolerance_m"]),
                        n_cars=int(n_cars), trial_seed=int(trial_seed))
                    rows.append({"status": "ok", "result": asdict(result)})
                except ValueError as error:
                    rows.append({"status": "failed", "scene": scene,
                                 "n_cars": int(n_cars), "trial_seed": int(trial_seed),
                                 "reason": str(error)})
    groups = defaultdict(list)
    for row in rows:
        item = row["result"] if row["status"] == "ok" else row
        groups[(item["scene"], item["n_cars"])].append(row)
    summary = []
    for (scene, n_cars), group in sorted(groups.items()):
        successes = [item["result"] for item in group if item["status"] == "ok"]
        summary.append({
            "scene": scene,
            "n_cars": n_cars,
            "trial_count": len(group),
            "successful_trials": len(successes),
            "failed_trials": len(group) - len(successes),
            "mean_safe_plan_count": (sum(item["safe_plan_count"] for item in successes) /
                                     len(successes) if successes else None),
            "mean_global_broken_contracts": (
                sum(item["global_broken_contracts"] for item in successes) /
                len(successes) if successes else None),
            "mean_adapted_broken_contracts": (
                sum(item["adapted_pairwise_broken_contracts"] for item in successes) /
                len(successes) if successes else None),
            "mean_solve_ms": (sum(item["solve_ms"] for item in successes) /
                              len(successes) if successes else None),
            "candidate_product": (successes[0]["enumerated_plan_count"]
                                  if successes else None),
        })
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config_snapshot.json", config)
    _write_json(output_dir / "trials.json", rows)
    _write_json(output_dir / "summary.json", summary)
    digest = hashlib.sha256()
    for relative in ("cooperation/physics.py", "cooperation/scene_extensions.py",
                     "cooperation/run_scene_extensions.py"):
        digest.update(relative.encode("utf-8"))
        digest.update((_ROOT / relative).read_bytes())
    _write_json(output_dir / "manifest.json", {
        "experiment": "E5_scene_specific_finite_candidate_sweep",
        "scenes": list(config["scenes"]),
        "vehicle_counts": list(config["vehicle_counts"]),
        "trial_seeds": list(config["trial_seeds"]),
        "total_trials": len(rows),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "scope": "new_provisional_kinematic_definitions_not_original_or_CARLA",
        "limits": [
            "Safety uses finite horizon center distances or lane headway, not vehicle-envelope certification.",
            "Lateral smoothstep has no steering or lateral acceleration bound.",
            "Welfare and contract distance are defined and compared within each scene only.",
            "Exactness holds only over each supplied finite action product.",
        ],
    })
    return output_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=_ROOT / "config" / "multicar_scene_extensions.json")
    parser.add_argument("--output-root", type=Path,
                        default=_ROOT / "Result" / "multicar_scene_extensions")
    args = parser.parse_args(argv)
    output_dir, summary = run_extension_sweep(args.config, args.output_root)
    print(json.dumps({"result_dir": str(output_dir), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
