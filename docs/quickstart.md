# Quick Start

完整命令手册见：`docs/中文操作手册.md`

## 1) 环境

建议先执行一键环境脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -VenvPath .venv_sim
```

GPU 环境建议：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -VenvPath .venv_gpu -TorchVariant auto -Recreate
```

激活后也可手动安装：

```powershell
python -m pip install -r requirements_sim.txt
```

## 2) 运行 2D 尖锐缺口算例（纯物理）

```powershell
python scripts/run_notched_case.py --config configs/notch_case.yaml --disable-ml --progress --progress-every 20
```

高精度细网格（GPU）：

```powershell
.\.venv_gpu\Scripts\python scripts/run_notched_case.py --config configs/notch_case_highres.yaml --progress --progress-every 20
```

## 3) 训练 surrogate

```powershell
python scripts/train_surrogate.py --config configs/notch_case.yaml
```

建议先用 `--disable-ml` 跑一段并设置 `--override-save-every 1`，可获得更稳定的训练样本。
建议改用物理时间参数：`--dt-s ... --total-time-s ... --save-interval-s ...`，其中 `--override-save-every` 仅作兼容。

## 4) 启用 ML predictor-corrector

```powershell
python scripts/run_notched_case.py --config configs/notch_case.yaml
```

## 5) 对比速度

```powershell
python scripts/benchmark_solver.py --config configs/notch_case.yaml --steps 200 --warmup-steps 20
```

## 6) 全流程检查（环境+测试+短程训练+基准）

```powershell
python scripts/run_all_checks.py --config configs/notch_case.yaml --device cuda --min-speedup 1.05 --max-solid-drift 0.1
```

## 7) 最终云图输出

每次仿真结束会自动输出：

- `final_clouds/von_mises_cloud.png`
- `final_clouds/yeta_cloud.png`
- `final_clouds/cMg_cloud.png`
- `final_clouds/phi_cloud.png`
