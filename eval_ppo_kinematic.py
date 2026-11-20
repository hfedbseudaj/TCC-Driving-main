
import os, sys, math, json, argparse, csv, hashlib
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import torch
import numpy as np
from opencood.models.comm_modules.rubinstein_bargaining import RubinsteinBargaining
from common.bargaining_evaluation import evaluate_policy
from common.experiment_protocol import environment_metadata, project_fingerprint, selection_score
from train_bargaining_ppo import StochasticPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260608
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SCENARIOS = {
    "intersection":{"a":{"start":[-30,-2],"target":[0,0],"speed":8},"b":{"start":[-2,-30],"target":[0,0],"speed":8}},
    "chase":{"a":{"start":[-25,0],"target":[15,0],"speed":10},"b":{"start":[-15,0],"target":[15,0],"speed":6}},
    "merge":{"a":{"start":[-25,-3],"target":[5,0],"speed":8},"b":{"start":[-30,0],"target":[10,0],"speed":7}},
    "head_on":{"a":{"start":[-20,0],"target":[10,0],"speed":8},"b":{"start":[20,0],"target":[-10,0],"speed":8}},
}

class SimState:
    def __init__(self, scenario="intersection", dt=0.1):
        cfg = SCENARIOS[scenario];self.dt=dt;self.name=scenario;self.step=0
        self.collision=False;self.finished=False
        a,b=cfg["a"],cfg["b"]
        self.pos=[np.array(a["start"],np.float32),np.array(b["start"],np.float32)]
        ta=np.array(a["target"],np.float32)-self.pos[0]
        tb=np.array(b["target"],np.float32)-self.pos[1]
        self.spd=[a["speed"],b["speed"]]
        self.hdg=[math.atan2(ta[1],ta[0]),math.atan2(tb[1],tb[0])]
        self._update_wps()
    def _update_wps(self):
        self.wps=[np.zeros((10,2),np.float32) for _ in range(2)]
        for v in range(2):
            for i in range(10):
                t=(i+1)*self.dt
                self.wps[v][i]=[self.spd[v]*t*math.cos(self.hdg[v]),self.spd[v]*t*math.sin(self.hdg[v])]
    def adjust(self,vid,dx,dy):
        spdc=dx*math.cos(self.hdg[vid])+dy*math.sin(self.hdg[vid])
        self.spd[vid]=max(0.3,min(15.0,self.spd[vid]+spdc*2.0))
    def step_sim(self):
        if self.finished:return
        for v in range(2):
            self.pos[v]+=np.array([self.spd[v]*self.dt*math.cos(self.hdg[v]),self.spd[v]*self.dt*math.sin(self.hdg[v])])
        self._update_wps();self.step+=1
        d=np.linalg.norm(self.pos[0]-self.pos[1])
        if d<1.0:self.collision=True;self.finished=True
        if self.step>200 or any(abs(p[0])>40 for p in self.pos) or any(abs(p[1])>40 for p in self.pos):
            self.finished=True
    def get_state(self):
        pos=torch.from_numpy(np.array(self.pos)).float().to(DEVICE)
        vel=torch.tensor([[self.spd[0]*math.cos(self.hdg[0]),self.spd[0]*math.sin(self.hdg[0])],[self.spd[1]*math.cos(self.hdg[1]),self.spd[1]*math.sin(self.hdg[1])]],dtype=torch.float32,device=DEVICE)
        wps=torch.from_numpy(np.array(self.wps)).float().to(DEVICE)
        return {"positions":pos,"velocities":vel,"waypoints":wps,"headings":torch.tensor(self.hdg,device=DEVICE),"record_len":torch.tensor([2],device=DEVICE)}

def run_episode(model, scenario, max_steps=60):
    sim = SimState(scenario); conv=0; wpds=[]
    while not sim.finished and sim.step < max_steps:
        s = sim.get_state()
        feat = torch.randn(2, 256, 48, 48, device=DEVICE)
        with torch.no_grad():
            out = model(spatial_features=feat, waypoints=s["waypoints"], positions=s["positions"], velocities=s["velocities"], headings=s["headings"], record_len=s["record_len"])
        wpd = (out["adjusted_waypoints"]-s["waypoints"]).norm(dim=-1).mean().item()
        wpds.append(wpd)
        if out["metadata"][0]["converged"]: conv+=1
        for v in range(2):
            if v<out["adjusted_waypoints"].shape[0]:
                dxy=(out["adjusted_waypoints"][v,0]-s["waypoints"][v,0]).cpu().numpy()
            else:
                dxy=-(out["adjusted_waypoints"][0,0]-s["waypoints"][0,0]).cpu().numpy()*0.5
            sim.adjust(v,dxy[0],dxy[1])
        sim.step_sim()
    n=max(1,sim.step)
    return {"steps":sim.step,"collision":sim.collision,"conv":round(conv/n*100,2)}

class ConstantDiscount(torch.nn.Module):
    def __init__(self, delta=0.9):
        super().__init__();self.delta=torch.nn.Parameter(torch.tensor(delta),requires_grad=False)
    def forward(self, ttc):return self.delta.expand_as(ttc)

class HandcraftedUtility(torch.nn.Module):
    def forward(self, state, offer):
        return torch.sigmoid(10.0*(offer[:,3]-0.5*offer[:,0])-2.0)

variants = [
    ("TTC + 3 rounds + a=0.3 (baseline)", {"max_rounds":3,"discount_type":"ttc","discount_alpha":0.3,"utility_type":"learned"}),
    ("Constant + 3 rounds + a=0.3", {"max_rounds":3,"discount_type":"constant","discount_alpha":0.3,"utility_type":"learned"}),
    ("TTC + 1 round + a=0.3", {"max_rounds":1,"discount_type":"ttc","discount_alpha":0.3,"utility_type":"learned"}),
    ("TTC + 3 rounds + a=0.1", {"max_rounds":3,"discount_type":"ttc","discount_alpha":0.1,"utility_type":"learned"}),
    ("TTC + 3 rounds + a=1.0", {"max_rounds":3,"discount_type":"ttc","discount_alpha":1.0,"utility_type":"learned"}),
]

def create_model(vcfg, checkpoint_state):
    cfg = {"state_dim":64,"offer_dim":4,"hidden_dim":128,"max_rounds":vcfg["max_rounds"],"discount_alpha":vcfg["discount_alpha"],"delta_min":0.05,"delta_max":0.99,"waypoint_offset_scale":2.0}
    model = RubinsteinBargaining(cfg).to(DEVICE)
    model.eval()
    if "model" not in checkpoint_state or "policy" not in checkpoint_state:
        raise RuntimeError("Expected a full PPO checkpoint containing 'model' and 'policy' state dictionaries")
    model.load_state_dict(checkpoint_state["model"], strict=True)
    if vcfg["discount_type"] == "constant":
        model.discount = ConstantDiscount(delta=0.9).to(DEVICE)
    if vcfg["utility_type"] == "handcrafted":
        model.utility_estimator = HandcraftedUtility().to(DEVICE)
    policy = StochasticPolicy(state_dim=64, offer_dim=4, hidden_dim=128).to(DEVICE)
    policy.load_state_dict(checkpoint_state["policy"], strict=True)
    policy.eval()
    return model, policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bargaining PPO policy in the kinematic simulator.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "eval_ppo_kinematic.yaml"), help="YAML config file")
    pre = parser.parse_known_args()
    try:
        from common.io import load_config_from_yaml
        cfg = load_config_from_yaml(pre[0].config) or {}
    except Exception as e:
        cfg = {}
        print(f"[Config] Warning: failed to load config: {e}")
    parser.add_argument("--checkpoint", default=cfg.get("checkpoint", None), help="Path to a trained PPO checkpoint.")
    parser.add_argument("--trials", type=int, default=cfg.get("trials", 25), help="Number of trials per scenario and variant.")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", SEED), help="Random seed for trial perturbation.")
    parser.add_argument("--training-seed", type=int, default=None, help="Training seed associated with this checkpoint.")
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default=None)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=Path("Result/ppo_kinematic_latest.json"))
    args = parser.parse_args()
    if args.checkpoint is None:
        print("Error: no checkpoint specified. Use --checkpoint <path> or set 'checkpoint' in config.")
        sys.exit(1)
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    trials = args.trials
    var_cfg = cfg.get("variants", None)
    if var_cfg:
        run_variants = [(v["name"], {k: x for k, x in v.items() if k != "name"}) for v in var_cfg]
    else:
        run_variants = variants
    print()
    print("=== PPO KINEMATIC VALIDATION (" + str(trials) + "x/scenario, seed=" + str(args.seed) + ") ===")
    print()
    hdr = "  {:<40} {:>5} {:>5} {:>5} {:>5} {:>8}".format("Variant", "Int", "Chse", "Mrg", "HOn", "Coll%")
    print(hdr)
    print("  " + "-" * 70)

    summary = {
        "seed": args.seed,
        "training_seed": args.training_seed,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "trials_per_scenario": trials,
        "scenario_order": list(SCENARIOS.keys()),
        "source_code": project_fingerprint(PROJECT_ROOT, [
            "eval_ppo_kinematic.py", "train_bargaining_ppo.py",
            "common/bargaining_evaluation.py", "common/experiment_protocol.py",
            "common/deterministic_bev.py", "common/kinematic_bargaining_env.py",
            "opencood/models/comm_modules/rubinstein_bargaining.py",
        ]),
        "environment": environment_metadata(torch, np),
        "evaluation_input": "common.deterministic_bev",
        "simulator": "common.kinematic_bargaining_env",
        "latency_budget_ms": args.latency_budget_ms,
        "variants": [],
    }
    all_episode_rows = []

    for name, vcfg in run_variants:
        model, policy = create_model(vcfg, checkpoint)
        results, metrics = evaluate_policy(
            model, policy, DEVICE,
            scenarios=list(SCENARIOS.keys()),
            trials_per_scenario=trials,
            seed_base=args.seed,
            max_steps=60,
            latency_budget_ms=args.latency_budget_ms,
        )
        for row in results:
            all_episode_rows.append({
                "variant": name,
                "training_seed": args.training_seed,
                "checkpoint_kind": args.checkpoint_kind,
                **row,
            })
        scenario_collisions = {
            sc: sum(bool(row["collision"]) for row in results if row["scenario"] == sc)
            for sc in SCENARIOS
        }
        line = "  {:<40} {:>5} {:>5} {:>5} {:>5} {:>7}%".format(
            name, scenario_collisions["intersection"], scenario_collisions["chase"],
            scenario_collisions["merge"], scenario_collisions["head_on"],
            round(100.0 * metrics["collision_rate"]))
        print(line)
        summary["variants"].append({
            "name": name,
            "config": vcfg,
            "collisions_by_scenario": scenario_collisions,
            "total_collisions": sum(scenario_collisions.values()),
            "total_trials": len(results),
            "collision_pct": 100.0 * metrics["collision_rate"],
            "near_collision_pct": 100.0 * metrics["near_collision_rate"],
            "mean_min_ttc_s": float(np.mean([row["min_ttc"] for row in results])),
            "mean_ttc_risk": metrics["mean_ttc_risk"],
            "convergence_pct": 100.0 * metrics["convergence_rate"],
            "mean_latency_ms": metrics["mean_latency_ms"],
            "selection_score": metrics["selection_score"],
        })

    print()
    print("  Legend: Int=Intersection, Chse=Chase, Mrg=Merge, HOn=Head-on")
    print("  Values are collisions out of " + str(trials) + " trials")

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  JSON -> " + str(out_path))
    episodes_path = out_path.with_name(out_path.stem + "_episodes.csv")
    if all_episode_rows:
        fields = [
            "variant", "training_seed", "checkpoint_kind", "scenario",
            "scenario_index", "trial_index", "trial_seed", "steps", "collision",
            "near_collision", "min_distance", "min_ttc", "ttc_risk", "conv",
            "latency_ms", "mean_waypoint_delta", "scene_perturbation",
        ]
        with episodes_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in all_episode_rows:
                output = dict(row)
                output["scene_perturbation"] = json.dumps(
                    output["scene_perturbation"], ensure_ascii=False
                )
                writer.writerow({key: output.get(key) for key in fields})
        print("  Episodes -> " + str(episodes_path))


if __name__ == "__main__":
    main()
