# TTC 校准的多车合作：研究原型与实验记录

本目录实现一条可核算的**多车合作研究链**：双车 TTC 折扣与 Rubinstein 份额 → 双车轨迹契约 `K_e` → 多辆 CAV 的联合动作选择。四类场景已有构造运动学闭环；交叉现统一支持 2/3/4 辆 CAV。E0–E5 见[首轮实验报告](reports/EXPERIMENT_REPORT_2026-11-21.md)，E6–E9 见[继续实验报告](reports/EXPERIMENT_REPORT_2026-11-21_E6_E9.md)，E10–E15 见[解码与安全实验报告](reports/EXPERIMENT_REPORT_2026-11-21_E10_E15.md)。最新空集回退、净距、E19–E23 分段制动核验与配对闭环结果见[E16–E23 复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md)。这些结果不等于 CARLA、公开数据集或真实车辆实验。

滚动控制器当前默认 `decoder_mode=coarse`、`joint_objective=contract_fidelity`、`welfare_epsilon_m=0.5`，预测额外净距为 1.0 m。交叉场景可试 `adaptive_crossing` 的 0.5 m/s² 双车解码网格，以及实验性 `share_then_contract` 选点；后者在固定安全与福利筛选下更直接惩罚边收益分配偏离，但需要更宽的福利容差才会影响动作，且已观察到延迟代价。默认值未因单一指标改动。

## 与原项目的关系

- 原论文的双车理论份额为 `δ_i = 1 - exp(-α_i · TTC)`，`s_A* = (1 - δ_B)/(1 - δ_A δ_B)`，`s_B* = δ_B(1 - δ_A)/(1 - δ_A δ_B)`。这里的 `s*` 是可分盈余的**份额**，不是速度或路点。
- 存档代码通过学习的四维报价、有限轮接受规则和路点调整产生动作，**没有直接把闭式 `s*` 解码为动作**。本目录新增的 `D₂` 是一个新定义的物理解码器，因此“双车退化”只指新方法内部的契约与动作回归；与存档代码的动作差距仍须单独测量。
- `compose.py` 的顺序比较器按决策前 TTC 排序，逐边提交 `K_e` 动作、覆盖共享车辆目标，再投影到共同安全/福利候选集；它是新接口上的**透明重建基线**，并非原学习报价代码或原作者多车程序的逐行复刻。任何优劣结论只能写成“与该重建基线比较”。

## 数学对象与代码对应

| 对象 | 目前定义 | 实现 |
| --- | --- | --- |
| `H_i(t_i)` | 车辆 `i` 以有界加减速度抵达冲突区、以指定速度通过的确定性短时轨迹 | `physics.py` 的 `VehicleSpec`、`canonical_approach_plan` |
| 双车候选 `p` | 两辆车各选一个进入时刻；占用窗不得重叠，至少留 `clearance_s` | `physics.py` 的 `pair_safe` |
| `g_i(p)` 与 `S_e(p)` | `g_i = t_i^0 - t_i(p)`，`S_e = g_i + g_j`，单位均为秒；`t_i^0` 是安全分歧方案进入时刻 | `crossing.py` |
| 份额 `r_i(p)` | `g_i / S_e`；只在 `g_i,g_j ≥ 0` 且 `S_e > 0` 时定义 | `crossing.py` |
| `D₂` | 先在安全、个体理性且 `S_e ≥ S_e^max - η` 的候选中筛选，再按份额偏差、负盈余、控制努力和固定 ID 字典序选唯一契约 | `decode_crossing_pair` |
| `K_e=(c_e,θ_e,C_e)` | 交叉场景中包含参考进入时刻、方向、容差和安全规则的轨迹对契约；方向由可执行候选决定 | `CrossingContract` |
| `F_N` | 所有提供的有限候选组合中，逐车轨迹可达且每条冲突边满足共同硬安全规则的集合 | `compose_crossing_snapshot` |
| `B(Π)` | 未兑现的边契约数；同一辆车关联多条边时逐边计数 | `compose.py` |
| `D_int(Π)` | 边契约参考时刻偏差的加权和，以 `t_ref_s` 归一化；一辆车在多条边上的偏差会分别计入，衡量**契约忠实度** | `compose.py` 的 `edge_fidelity` |
| `D_vehicle(Π)` | 每辆车相对单一车辆参考时刻的偏差，仅作为另报的按车指标 | `compose.py` 的 `vehicle_change` |
| `W_int(Π)` | `-Σ_i [t_i - d_i/v_i]`，即自由流进入时刻以上的总延误取负，单位秒 | `compose.py` 的 `welfare_s` |

全局选择先计算有限候选内的 `W* = max_{Π∈F_N} W_int(Π)`，仅保留 `W_int(Π) ≥ W* - ε_W` 的计划，再按 `(B, D_int, -W_int, 固定决胜键)` 做字典序最小化。因此“最小修复”指**给定有限候选族和福利底线**下的精确枚举结果，不是连续动作空间中的全局最优证明。`B=0` 表示该候选兑现全部边契约；`D_int` 不是碰撞率或总延误。

## 人为构造示例

配置见 [`config/multicar_cooperation_example.json`](../config/multicar_cooperation_example.json)。三个 CAV 形成 `A—B—C` 两条冲突边；A 与 C 在平行路径上，B 横穿两条路径。初态、参数、候选时刻和 `B—C` 契约均为**手工教学示例，未做数据校准**；决策前 TTC 则由这些构造初态按旧代码的闭合速度公式计算。`A—B` 得 `TTC=3 s`，取 `α_A=α_B=0.2 s⁻¹`，得到 `δ≈0.4512`，`s_A*≈0.6891`。两个安全候选为 `(t_A,t_B)=(3.0,5.8)` s 与 `(5.2,3.0)` s；在 `η=1.0 s` 的示例设置下，新解码器选择前者。`B—C` 人工契约参考 `(t_B,t_C)=(7.0,4.0)` s，使 B 在两条边上收到不相交的参考时刻。这个冲突只验证“局部协议可能需要合成/修复”的接口，不是对原项目碰撞原因的证据。

车辆初态、加减速限值、有效占用长度及所有候选时刻均在 JSON 中逐项给出。当前轨迹生成器已检查这些候选在**有界纵向运动模型**中可达；有效占用长度用于简化的时窗安全检查，尚未建立车辆几何包络、真实地图车道或闭环控制。旧闭合速度 TTC 在本例中给出 `A—B=3.0 s`、`B—C≈3.7243 s`；它不是按冲突区占用预测得到的 TTC，不能代替时窗安全检验。

最新 E2/E3a 快照中，45 个候选组合有 11 个满足简化硬安全条件，没有候选能同时兑现两条边契约。全局与 TTC 顺序重建都选 `(t_A,t_B,t_C)=(3,7,4)` s，违约 1 条边、福利 `−4.0 s`、边契约偏差 `1.2`；快照没有显示全局法优势。该例中的 `B—C` 契约是人为给定的，只用于记录共享车辆目标被后续边覆盖及联合筛选，不是原模型多车碰撞原因的证据。

## 运行与记录

在项目根目录、项目自己的 Python 3.7 虚拟环境中运行：

```powershell
.\.venv\Scripts\python.exe -m cooperation.run_snapshot --config config/multicar_cooperation_example.json --allow-constructed-example
.\.venv\Scripts\python.exe -m cooperation.archived_audit --max-steps 80
.\.venv\Scripts\python.exe -m cooperation.run_paired_closed_loop --allow-constructed-example
.\.venv\Scripts\python.exe -m cooperation.run_adapter_audit --allow-constructed-example
.\.venv\Scripts\python.exe -m cooperation.run_adapter_audit --allow-constructed-example --allow-safety-projection
.\.venv\Scripts\python.exe -m cooperation.run_scene_extensions
.\.venv\Scripts\python.exe -m cooperation.run_scene_closed_loop --trial-seeds 301 302 303 --n-cars 2 3 4
.\.venv\Scripts\python.exe -m cooperation.run_mechanism_ablation --trial-seeds 301 302 303 --n-cars 2 3 4
.\.venv\Scripts\python.exe -m cooperation.run_scene_closed_loop --trial-seeds 301 302 303 --methods global sequential_pairwise_reconstruction welfare_only clearance_first
.\.venv\Scripts\python.exe -m cooperation.run_fixed_candidate_audit --n-cars 2 3 4 --trial-seeds 301 302 303
.\.venv\Scripts\python.exe -m cooperation.run_scene_closed_loop --scenes crossing --n-cars 2 3 4 --trial-seeds 301 302 303 --methods global --decoder-mode adaptive_crossing --joint-objective share_then_contract --welfare-epsilon-m 2.0
.\.venv\Scripts\python.exe -m cooperation.run_e24_broad_staged_recovery_paired --generate-stimuli
.\.venv\Scripts\python.exe -m cooperation.run_e24_broad_staged_recovery_paired --compact
.\.venv\Scripts\python.exe -m cooperation.run_e25_staged_recovery_scope_audit
.\.venv\Scripts\python.exe -m cooperation.run_e26_cross_scene_staged_candidate_audit
.\.venv\Scripts\python.exe -m cooperation.run_e27_non_chase_staged_counterexamples --compact
.\.venv\Scripts\python.exe -m cooperation.run_e28_non_chase_neighborhood_scan --compact
.\.venv\Scripts\python.exe -m cooperation.run_e29_non_chase_staged_recovery_paired --generate-stimuli
.\.venv\Scripts\python.exe -m cooperation.run_e29_non_chase_staged_recovery_paired --compact
.\.venv\Scripts\python.exe -m cooperation.run_e30_chase_mismatch_and_holdout_audit --generate-stimuli
.\.venv\Scripts\python.exe -m cooperation.run_e30_chase_mismatch_and_holdout_audit --compact
.\.venv\Scripts\python.exe -m cooperation.run_e31_four_scene_tail_latency_audit --compact
.\.venv\Scripts\python.exe -m cooperation.run_e32_alpha_and_weight_sensitivity --generate-stimuli
.\.venv\Scripts\python.exe -m cooperation.run_e32_alpha_and_weight_sensitivity --compact
.\.venv\Scripts\python.exe -m cooperation.run_e33_final_four_method_comparison --compact
.\.venv\Scripts\python.exe -m cooperation.run_e34_legacy_offer_priority_bridge_audit --compact
```

运行器要求显式的 `--allow-constructed-example`，避免把教学快照误认作正式实验。结果写入 `Result/multicar_cooperation/<run_id>/`：`config_snapshot.json`、`pair_contracts.json`、`cooperation_snapshot.json`、`selected_trajectory_samples.json` 和 `manifest.json`。清单记录代码与配置哈希、Git 提交、Python/平台版本、单次求解时间、`train_seed` 与 `trial_seed`。此例没有训练，`train_seed=null`；没有随机场景采样，`trial_seed` 仅为记录字段。配置中的 `dt_s` 和 `horizon_s` 用于采样选中方案的运动学参考轨迹，不参与有限候选求解，也不构成车身碰撞检测或控制器执行；`α_C` 因 `B—C` 是人工契约而未参与该边解码。单次快照耗时不构成实时性证据，也不报告碰撞率。

## 实施状态与下一步

| 范围 | 当前状态 |
| --- | --- |
| 交叉场景 `D₂`、可达轨迹、双车契约 `K_e` | 已有有限候选实现；仍需系统验证候选生成、份额有效性与双车动作回归 |
| 三 CAV 两边的合作合成与有限候选修复 | 已有单次快照实现，属于 E2/E3a 的最小原型 |
| E0 原代码行为复核 | `archived_audit.py` 只读加载存档模块，在原四类双车场景记录报价、TTC、折扣、动作和轨迹；已用 best 检查点与同 seed 随机初始化在 RTX 3090 做 19 个场景 trial/场景。追车和对向接近暴露高碰撞率；存档没有作者的 3/4 车脚本，不能由双车审计归因旧论文多车碰撞 |
| E1 双车解码器审计 | `run_decoder_audit.py` 对四种场景各跑 19 个扰动 seed，核验候选零距集合、双车内部回归、安全性与份额误差；并线份额误差仍偏高 |
| E2/E3a 快照 | 三车两边的单次手工快照已记录候选数、边目标覆盖与全局/顺序结果；两者最后选到相同方案，因此不构成全局法优越性证据 |
| E3b 配对闭环 | `crossing_closed_loop.py` 每周期重算交叉场景候选、边契约和联合可行域；无可行计划时用一步制动/巡航/加速恢复器。19 个配对 trial 中两法均完成、无模型碰撞；恢复器不是形式安全证明 |
| E4 旧四维学习报价适配器 | `legacy_adapter.py` 固定映射“旧报价→旧路点头→旧速度命令→新物理可达轨迹”，`run_adapter_audit.py` 逐边记录严格适配失败率；可显式开启到同一安全候选族的投影，并报告投影幅度。适配器是新增接口，结果不能直接归因于旧学习模块 |
| E5 四类场景闭环 | `scene_closed_loop.py` 每 0.2 s 重规划 2 s 滚动预测，限制纵横向加速度/速度，使用有余量的矩形包络并统计闭环指标。历史 30 个配对种子覆盖追车、并线、对向 2/3/4 车与交叉 4 车，共 10 组、600 个 episode；当前代码已补交叉 2/3 车与全车安全分歧/个体收益检查，见 E6–E9 |
| E6–E9 机制与基线 | 同一闭环求解器覆盖四类场景 2/3/4 车；配对比较 TTC/固定折扣/均分和全局/顺序/仅福利/净距优先。TTC 份额变化多数没有传到最终动作，全局方法尚未显示稳定增益，详见继续实验报告 |
| E10–E15 解码与安全 | 交叉增加有界细解码与固定候选集审计；修复保持车道目标，补齐追车合法加速档，按完整余量、物理包络、未认证三层记录空集回退。四场景独立种子与实验性份额优先目标均已测试，详细正负结果见解码与安全实验报告 |
| E16–E18 回退与净距复核 | 预测/执行扫掠净距改为解析求解，回退增加来源、当前余量与分段候选日志，并新增初态相对制动余量诊断。E18 本机与服务器 Python 3.7.12 同参数各跑 240 个 episode，均出现同一追车碰撞；服务器最大单次求解 192.9 ms，分段制动本批未触发。历史结果见 E16–E18 报告 |
| E19 动态制动余量筛选 | 增加可选 3σ 启发式制动余量筛选与初态低余量标记，并保留全部种子。追车三车 20 个配对种子中，基线和筛选均在 seed 820、0.8 s 碰撞；筛选把回退从 15 增至 26 次，暂未带来安全收益，因此默认关闭。详见 [E16–E23 复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md) 和 `Result/ttc_multicar_20261121/E19_braking_filter_pair_local/`、`E19_braking_filter_server/` |
| E20 分段制动候选最小核验 | 一个固定三车追车状态下，恒定动作族 100 个联合候选无一满足 1 m 额外预测净距；分段恢复族 648 个候选中有 1 个满足，控制器实际选择该候选。本机 Python 3.7.9 和服务器 Python 3.7.12 的结果与代码哈希一致，服务器定向测试 19 项通过。该结果只说明有限动作族在单一无噪声快照中的差异，不是闭环安全收益。入口为 `run_e20_staged_braking_counterexample.py`，数据位于 `Result/ttc_multicar_20261121/E20_staged_braking_counterexample/`，详见 [E16–E23 复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md) |
| E21 分段动作族邻域网格扫描 | 围绕 E20 状态预先固定 3⁵=243 个微小位置/速度偏移状态；恒定动作族 68/243 个状态有合格候选，分段恢复族 112/243 个状态有合格候选，其中 44 个为分段族新增、131 个两类都失败。本机 Python 3.7.9 与服务器 Python 3.7.12 的逐状态结果解析后完全一致，服务器定向测试 19 项通过。该扫描是局部确定性网格，不是交通分布估计或闭环安全证据。入口为 `run_e21_staged_braking_neighborhood_scan.py`，逐状态记录和 manifest 位于 `Result/ttc_multicar_20261121/E21_staged_braking_neighborhood_scan/`，详见 [E16–E23 复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md) |
| E22 一般扰动配对闭环 | 追车三车 20 个新种子、位置/速度扰动分别为 1.0 m/0.8 m/s；两臂均未调用分阶段动作，碰撞 0、近碰撞各 1、完成 20/20。机制未触发，不能据此判定动作族有效或无效。详见 `Result/ttc_multicar_20261121/E22_staged_recovery_paired/` 与 [E16–E23 复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md) |
| E23 目标状态配对闭环 | 围绕 E20 状态的小扰动压力测试，20 个种子两臂均完成且无碰撞/近碰撞；处理臂 20/20 场景实际触发分段动作，共 37 步，但延迟约增加 0.00667 s，平均净距与对照近似。本机和服务器逐种子闭环指标一致，服务器使用既有 Python 3.7.12 环境，定向测试 19 项通过。仅为构造边界附近证据，不代表一般交通安全收益。入口 `run_e23_staged_recovery_targeted_paired.py`，完整数据在 `Result/ttc_multicar_20261121/E23_staged_recovery_targeted_paired/` 和 `E23_staged_recovery_targeted_paired_server/` |
| E24 固定扰动宽范围追车配对闭环 | 围绕 E20 初态做 `3^5=243` 个位置/速度状态，执行噪声先写入固定 JSON 并由本机、服务器逐字节共用。两臂均完成 243/243、碰撞 0、近碰撞各 26 且逐状态一致；分段动作在处理臂 17 个状态触发 22 步，未减少近碰撞，求解时间略增。完整轨迹均保留；忽略平台相关求解计时字段后，跨主机 episode 记录一致。详见 [E24–E26 报告](reports/EXPERIMENT_REPORT_2026-11-21_E24_E26.md)。 |
| E25 其他三类代码范围审计 | 交叉、并线、对向分别检查 2/3/4 车组，共 27 次动作接口检查和 9 个确定性基线 episode。历史实现仅在追车启动恢复搜索；其他三类当时未接入动作族。此结果说明当时“未实现”，不说明物理上不适用。 |
| E26 跨场景候选快照探查 | 交叉、并线、对向各用一个 2/3/4 车标准快照，把分段制动候选临时加入相同安全联合筛选器。九组普通动作原本都有安全候选；交叉 3/4 车快照选中了分段动作，并线/对向没有选中。仅为候选层探查，不执行闭环、不报告安全收益；对向 4 车候选数增加至 9,690，需关注联合枚举开销。 |
| E27 非追车三类场景边界反例核验 | 为垂直交叉、并线、对向接近分别构造初始净距达标但恒定动作族安全候选为空集（0 个）的三车边界状态；接入按需触发的场景专属分段恢复动作族后，达标候选分别恢复为 1、2、6 个且均被控制器选中。本机与服务器结果及哈希完全一致，详见 [E27–E29 报告](reports/EXPERIMENT_REPORT_2026-11-22_E27_E29.md)。 |
| E28 非追车三类场景 243 状态邻域网格扫描 | 围绕 E27 三个边界状态各做 `3^5=243`（合计 729 个状态）局部网格扫描，保留全部失败状态。分段恢复族在交叉新增 36/243、并线新增 47/243、对向新增 143/243 个可行状态，本机与服务器逐状态解析结果完全一致。 |
| E29 冻结共享扰动的非追车配对闭环 | 交叉、并线、对向各 20 个边界微扰种子（共 60 对），初态与 80 步执行加速度噪声冻结于同一 JSON 并由本机与服务器共用。60/60 个处理臂 episode 实际触发分段恢复；并线与对向的未认证下一步回退分别由 10→0 和 70→0，总回退由 55→20 和 236→60，平均最小净距与延迟同步改善；跨主机逐状态闭环指标完全一致。详见 [E27–E29 报告](reports/EXPERIMENT_REPORT_2026-11-22_E27_E29.md)。 |
| E30 追车预测-扰动失配分解与独立种子筛选复核 | 定量分解 `seed 820` 无噪声极限规避净距（`0.8 s` 仅 `+0.03332 m`）与实际高斯噪声累积损失（`-0.03742 m`）导致的 `-0.00410 m` 碰撞；在全新独立种子 `901–920`（追车 2/3/4 车共 60 对冻结扰动 episode）双端复核 `3σ` 筛选器，确认其不减碰撞反使 2 车回退 `21→83`、耗时翻倍，正式维持关闭/撤下。详见 [E30–E31 报告](reports/EXPERIMENT_REPORT_2026-11-22_E30_E31.md)。 |
| E31 四车联合候选枚举压缩与 240-episode 尾部时延复核 | 引入无损增量 DFS 安全剪枝、单遍配对净距查表复用与惰性动态余量计算。在与 E18 相同的 240 个 episode 上双端行为 100% 等价；服务器全部 12 组最大 P95 降至 `21.18 ms`（较 E18 下降 50.1%），对向/交叉/追车/并线 4 车服务器平均耗时分别降至 `13.38 / 4.30 / 4.43 / 2.61 ms`。详见 [E30–E31 报告](reports/EXPERIMENT_REPORT_2026-11-22_E30_E31.md)。 |
| E32 $\alpha$ 与边权度数 $w_e$ 敏感性扫描 | 在冻结共享扰动（种子 921–930）上扫描 $\alpha \in \{0.1, 0.2, 0.3, 0.5\}\text{ s}^{-1}$ 与三种边权归一化（`uniform`、`inverse_max_degree`、`inverse_mean_degree`）。全部 100% 完成、0 碰撞；$\alpha=0.2/0.3$ 违约边数最低，三种度数权重选出的联合轨迹完全一致。详见 [E32–E34 报告](reports/EXPERIMENT_REPORT_2026-11-22_E32_E34.md)。 |
| E33 最终版控制器四方法完整闭环对照 | 在集成全场景分段恢复与 DFS 剪枝的最终控制器上，对 4 场景 × 2/3/4 车 × 10 种子对比 `global`、`sequential`、`welfare_only` 与 `clearance_first`（共 480 episodes）。四法均 120/120 完成、0 碰撞；`global` 平均违约边数最低（`1.1687`），服务器平均求解仅 `3.50 ms`。详见 [E32–E34 报告](reports/EXPERIMENT_REPORT_2026-11-22_E32_E34.md)。 |
| E34 旧学习报价有界优先级桥接对比 | 使用训练 seed 608 的 `best_model.pth`，在 E4 的 10 个三车交叉样本上对比新 `priority_bounded_bridge` 与原 `naive_safety_projected`。新桥接器 20/20 边合法，平均 L1 修正从 `2.5295 s` 降至 `0.2559 s`（降幅 89.9%），违约边数由 `1.0` 降至 `0.6`。详见 [E32–E34 报告](reports/EXPERIMENT_REPORT_2026-11-22_E32_E34.md)。 |

### E5 当前专用定义

| 场景 | 新设计的动作与分歧 | 硬筛选与场景距离 |
| --- | --- | --- |
| 同向追车 | 后车有限制动/巡航，前车巡航/加速；安全分歧为后车强制动、前车巡航 | 每个采样时刻前后车间距至少 5 m；契约距离按纵向加速度差计算 |
| 并线 | 并线车用 3 s 五次平滑横移 3 m，并选纵向加速度；分歧为取消并线并制动 | 每步扫掠中心距离至少 2.2 m；契约距离分别计纵向加速度与横移量偏差 |
| 对向接近 | 各车选纵向制动/巡航和 −3/0/+3 m 横移；分歧为双方在会车前制动 | 每步扫掠中心距离至少 2.2 m；契约距离分别计加速度与横移量偏差 |

三类场景均在 6 s、0.1 s 步长的有限轨迹族中计算 `g_i=u_i(p)-u_i(p⁰)`、`S_e=g_i+g_j`，按原 `s*` 选择各自 `K_e`，再于本场景的安全联合候选中按违约数、边契约距离、场景福利选择。效用以 6 s 内前进距离减控制努力为基础；并线另有完成奖励。具体控制集合和数值在 `scene_extensions.py`，实验配置在 `config/multicar_scene_extensions.json`。这些都是**新建的待验证定义**，不属于原论文的报价或真实车辆动力学。

### E5 滚动闭环模型与结果

滚动闭环由 `scene_closed_loop.py` 实现，独立于上面的静态快照定义。它每 0.2 s 在有限动作族上重算 2 s 预测；场景总时限 16 s。安全检查以 4.5 m × 1.8 m 车辆外形并加 0.5 m 纵向、0.2 m 横向余量构造轴对齐包络，并对预测与执行路段细分采样。追车与并线使用同向车辆的纵向间距；对向场景为同车道相向接近，车辆可制动或向相邻车道避让。所有结果都使用手工构造的初态和简化控制器。

运行器 `run_scene_closed_loop.py` 使用 30 个场景 trial 种子，配对运行全局契约修复与 TTC 顺序两两重建。当前包含追车、并线、对向 2/3/4 车和交叉 4 车，共 10 组、600 个 episode。最新单次枚举版本见 [E5 摘要](../Result/ttc_multicar_20261121/E5_closed_loop_single_pass/20261121T014608Z_15f0676f/summary.json)、[运行清单](../Result/ttc_multicar_20261121/E5_closed_loop_single_pass/20261121T014608Z_15f0676f/manifest.json) 与[压缩轨迹](../Result/ttc_multicar_20261121/E5_closed_loop_single_pass/20261121T014608Z_15f0676f/episodes.json.gz)。所有组 30/30 完成、未观察到模型碰撞；四车交叉每种方法各有 3 次近碰撞。结果只对应构造运动学测试分布。

在这些构造数据上，全局方法与顺序重建基线的完成率和延迟接近，没有显示稳定优势；多车过滤前经常出现不安全原始联合目标，闭环共同安全筛选将其挡下。一次枚举减少了全局四车对向组的平均求解时间（40.60→35.23 ms）和 P95（55.09→48.32 ms），但最终观测最大值仍有 548.09 ms（全局四车交叉）和 472.36 ms（顺序四车对向），超过 200 ms 控制周期。当前只能说多数规划周期较快，不能声称硬实时；该结果也不代表 CARLA 或实车性能。

### E3b 时刻承诺权重敏感性

`crossing_closed_loop.py` 的 `commitment_weight` 用于惩罚本周期目标时刻相对上一周期承诺的变化。对权重 0、1、5 分别使用 101–110 的 10 个探索 trial 和 201–220 的 20 个新增 trial 做配对比较。权重 0 在 30 个 trial 中两种方法均 30/30 完成、0 次碰撞；权重 1 的全局法 29/30 完成并有 1 次碰撞，顺序法 30/30；权重 5 的全局法 28/30、顺序法 24/30，并出现碰撞。权重提高时过滤前冲突与恢复动作通常更多，时延收益并不一致。因此当前默认 `commitment_weight=0`，该项保留用于敏感性实验，不把它当成已验证的改进机制。摘要分别保存在 `Result/multicar_commitment_sensitivity/` 和 `Result/multicar_commitment_holdout/`。

E3a 只在**同一决策时刻**固定边契约、候选族和安全规则后替换聚合器，不执行车辆、不报告碰撞率；E3b/E5 报告闭环结果。所有本轮运行都保存了模型/场景 seed、代码或配置哈希、环境版本、summary 和 trial/episode 原始记录。它们仍是构造运动学证据，不能推断实车安全或方法优越性；不同场景的收益与距离不跨场景求和。E16–E23 净距、回退和分段候选核验结果见 [本轮复核报告](reports/EXPERIMENT_REPORT_2026-11-21_E16_E23.md)；更早完整实验与失败案例见 [2026-11-21 实验报告](reports/EXPERIMENT_REPORT_2026-11-21.md)。2026-11-20 的旧 E5 碰撞统计使用了错误的步间扫掠端点，不用于碰撞结论，旧文件仍保留作历史记录。
