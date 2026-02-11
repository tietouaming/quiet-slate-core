# 镁合金孪晶-晶体塑性-腐蚀耦合相场程序说明

## 已实现模块

- 腐蚀相场 `phi`（Allen-Cahn）
- Mg 离子扩散 `c`（含界面耦合项）
- 孪晶序参量 `eta`（TDGL）
- 晶体塑性滑移（12 系统，幂律滑移 + 潜硬化）
- 小变形力学平衡 `∇·σ=0`
- 机械加速腐蚀迁移率（按 Kovacevic 公式）
- 2D 微米级尖锐缺口几何初始化
- ML surrogate（predictor-corrector）
- 电场/氢脆耦合插件接口（可开关）
- 数值稳定化（过应力滑移、硬化速率上限、应变/位移限幅、surrogate 门控）
- surrogate 训练支持按快照步长自动换算为“每步残差”，避免 `save_every>1` 导致的多步漂移
- 运行时默认清理旧快照与旧图像，避免历史文件污染训练集

## 文件结构

- `mg_coupled_pf/config.py`: 参数与配置读取
- `mg_coupled_pf/simulator.py`: 耦合时间推进主求解器
- `mg_coupled_pf/crystal_plasticity.py`: CP 模块
- `mg_coupled_pf/mechanics.py`: 力学平衡
- `mg_coupled_pf/couplers.py`: 电场/氢脆扩展接口
- `mg_coupled_pf/ml/surrogate.py`: ML surrogate
- `scripts/run_notched_case.py`: 运行算例
- `scripts/train_surrogate.py`: 训练 surrogate
- `scripts/benchmark_solver.py`: 速度基准测试

## 快速运行

```powershell
.\.venv_formula\Scripts\python scripts/run_notched_case.py --config configs/notch_case.yaml --dt-s 5e-5 --total-time-s 0.01
```

GPU 运行示例：

```powershell
.\.venv_gpu\Scripts\python scripts/run_notched_case.py --config configs/notch_case.yaml --device cuda --mixed-precision --dt-s 5e-5 --total-time-s 0.01
```

输出目录默认：

- `artifacts/sim_notch/mg_notch_2d/snapshots`
- `artifacts/sim_notch/mg_notch_2d/figures`
- `artifacts/sim_notch/mg_notch_2d/history.csv`

## 训练 ML surrogate

```powershell
.\.venv_formula\Scripts\python scripts/train_surrogate.py --config configs/notch_case.yaml
```

训练完成后模型保存到：

- `artifacts/ml/surrogate_latest.pt`

再次运行仿真时会自动读取该模型并启用 `predictor-corrector`（若 `ml.enabled=true`）。

## 基准测试

```powershell
.\.venv_gpu\Scripts\python scripts/benchmark_solver.py --config configs/notch_case.yaml --device cuda --steps 100 --warmup-steps 20
```

返回纯物理与 ML 启用的步速对比。  

## 全流程自检

```powershell
.\.venv_formula\Scripts\python scripts/run_all_checks.py --config configs/notch_case.yaml
```

该脚本会自动执行 `pytest`、短程纯物理仿真、surrogate 训练、ML 短程仿真和 benchmark，并输出报告 JSON。  
