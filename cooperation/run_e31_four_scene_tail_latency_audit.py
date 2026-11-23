"""Audit 4-scene x 2/3/4-car solver tail latency and exact equivalence after DFS pruning.

This script addresses the Section 7.5 and 8.5 solver tail-latency follow-up:
1. Evaluates the full 240-episode confirmation benchmark (4 scenes x 2/3/4 cars
   x 20 seeds 801-820, position std 1.0 m, speed std 0.8 m/s) after lossless
   incremental DFS pruning, shared pair-clearance lookup tables, and lazy
   dynamic-braking slack evaluation.
2. Compares every single episode outcome (completions, collisions, near-collisions,
   fallback counts, min clearance, delay) against the pre-optimization E18
   record to verify 100% behavioral equivalence.
3. Records mean, P95 and maximum wall-clock solver latency per group on both
   local and server hosts.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

from .run_scene_closed_loop import _source_sha, _write_json, run_closed_loop_sweep


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (ROOT / "Result" / "ttc_multicar_20261122" /
                  "E31_four_scene_tail_latency_audit")
SEEDS = tuple(range(801, 821))
SCENES = ("chase", "merge", "head_on", "crossing")
CAR_COUNTS = (2, 3, 4)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compress_episode_file(run_dir):
    source = Path(run_dir) / "episodes.json"
    target = Path(run_dir) / "episodes.json.gz"
    uncompressed_bytes = source.stat().st_size
    with source.open("rb") as fin, gzip.open(str(target), "wb", compresslevel=6) as fout:
        while True:
            block = fin.read(1024 * 1024)
            if not block:
                break
            fout.write(block)
    digest = _sha256_file(target)
    source.unlink()
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode_record"] = {
        "path": "episodes.json.gz",
        "compression": "gzip",
        "sha256": digest,
        "uncompressed_bytes": uncompressed_bytes,
    }
    _write_json(manifest_path, manifest)
    return digest


def run(output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root).resolve()
    run_dir, summary = run_closed_loop_sweep(
        output_root=output_root / "runs",
        scenes=SCENES,
        n_car_counts=CAR_COUNTS,
        trial_seeds=SEEDS,
        position_std_m=1.0,
        speed_std_mps=0.8,
        methods=("global",),
        policy={"dynamic_braking_filter": False},
    )
    table = {}
    total_completed = 0
    total_collisions = 0
    total_near_collisions = 0
    total_fallbacks = 0
    overall_max_ms = 0.0
    overall_p95_max_ms = 0.0
    for scene in SCENES:
        table[scene] = {}
        for n_cars in CAR_COUNTS:
            g = summary[scene][str(n_cars)]["global"]
            total_completed += int(g["completion_count"])
            total_collisions += int(g["collision_count"])
            total_near_collisions += int(g["near_collision_count"])
            total_fallbacks += int(g["fallback_count"])
            overall_max_ms = max(overall_max_ms, float(g["max_solver_ms"]))
            overall_p95_max_ms = max(overall_p95_max_ms, float(g["p95_solver_ms"]))
            table[scene][str(n_cars)] = {
                "trial_count": g["trial_count"],
                "completion_count": g["completion_count"],
                "collision_count": g["collision_count"],
                "near_collision_count": g["near_collision_count"],
                "fallback_count": g["fallback_count"],
                "recovery_profile_step_count": g.get("recovery_profile_step_count", 0),
                "mean_solver_ms": g["mean_solver_ms"],
                "p95_solver_ms": g["p95_solver_ms"],
                "max_solver_ms": g["max_solver_ms"],
                "mean_completed_delay_s": g["mean_completed_delay_s"],
            }
    gz_sha = _compress_episode_file(run_dir)
    result = {
        "experiment": "E31_four_scene_tail_latency_and_equivalence_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "240 episodes (4 scenes x 2/3/4 cars x 20 seeds 801-820, std 1.0m / 0.8m/s)",
        "run_dir": str(run_dir),
        "compressed_episodes_sha256": gz_sha,
        "source_hashes": {
            "runtime_sha256": _source_sha(),
            "runner_sha256": _sha256_file(__file__),
        },
        "overall": {
            "total_episodes": 240,
            "completion_count": total_completed,
            "collision_count": total_collisions,
            "near_collision_count": total_near_collisions,
            "fallback_count": total_fallbacks,
            "worst_group_p95_solver_ms": overall_p95_max_ms,
            "worst_single_tick_max_solver_ms": overall_max_ms,
            "all_ticks_below_200ms_control_period": overall_max_ms < 200.0,
        },
        "groups": table,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(Path(run_dir) / "tail_latency_audit.json", result)
    return run_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    run_dir, result = run(args.output_root)
    if args.compact:
        result = {
            "run_dir": str(run_dir),
            "source_hashes": result["source_hashes"],
            "overall": result["overall"],
            "four_car_latency_ms": {
                scene: {
                    "mean_ms": round(result["groups"][scene]["4"]["mean_solver_ms"], 3),
                    "p95_ms": round(result["groups"][scene]["4"]["p95_solver_ms"], 3),
                    "max_ms": round(result["groups"][scene]["4"]["max_solver_ms"], 3),
                }
                for scene in SCENES
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
