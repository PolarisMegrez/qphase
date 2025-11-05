# QPhaseSDE 用户指南（v0.1.3）

目标读者：具备基础 Python 知识的物理学研究者，希望进行复数模态的随机微分方程仿真与可视化。

## 安装（Windows PowerShell）

推荐使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e packages\QPhaseSDE
pip install -e packages\QPhaseSDE_cli
python -m pip install ruamel.yaml
```

## 第一次运行

使用示例模型（双模 Van der Pol）与配置：

```powershell
qps run sde --config configs\vdp_run.yaml
```

将生成 `runs/` 下的新目录，包含：
- `config_snapshot/` —— 运行时使用的配置快照
- `time_series/` —— 每个 IC 的时序 NPZ
- `figures/` —— 每个 IC 的图片
- `manifest.json` —— 元数据

提示：`model.ic` 可为单个向量或多个向量；多个 IC 会被逐一独立仿真。

## 可视化

内置两类图形，可在同一次运行中请求多张图：

- 相图（`run.visualizer.phase_portrait`）：
  - `kind: re_im` 且 `modes` 含 1 个索引 → 绘制 Re(α) vs Im(α)
  - `kind: abs_abs` 且 `modes` 含 2 个索引 → 绘制 |α_i| vs |α_j|
  - 可选 `t_range: [t_start, t_end]` 指定时间窗口
  - 样式位于 `profile.visualizer.phase_portrait.{re_im|abs_abs}`

- 功率谱密度（PSD，`run.visualizer.psd`）：
  - `kind: complex`（对复信号 FFT）或 `modular`（对 |信号| FFT）
  - `modes: [...]` —— 一张图可包含多个模式
  - 可选 `xlim: [xmin, xmax]` 与 `t_range: [t_start, t_end]`
  - 全局样式位于 `profile.visualizer.psd`：
    - `convention: symmetric|unitary|pragmatic`
    - `x_scale: linear|log`, `y_scale: linear|log`
  - 数据对轨迹求平均。`complex` 采用双边；`modular` 采用单边显示。

## 保存开关与容量保护

位于 `profile.save`：
- `save_timeseries`：保存每个 IC 的时序 NPZ
- `save_psd_complex`：保存复信号 PSD NPZ
- `save_psd_modular`：保存模值 PSD NPZ

为避免磁盘暴涨，CLI 在保存时序前会估算占用，若预计超过 1 GiB，默认中止。可使用 `--max-storage-gb` 提升上限（谨慎使用）。

## 基于已有结果重绘

无需重新积分即可重绘图片：

```powershell
qps analyze phase --from-run runs\<run_id>
qps analyze psd --from-run runs\<run_id>
```

也可临时覆盖配置（以相图为例）：

```powershell
qps analyze phase --from-run runs\<run_id> --use-snapshot false --specs-json "[{\"kind\":\"re_im\",\"modes\":[0]}]"
```

## 故障排查

- 缺少 YAML 解析：安装 `ruamel.yaml` 或 `PyYAML`。
- 磁盘占用大：增大 `save_every`，关闭 `save_timeseries`，或确认后提升 `--max-storage-gb`。
- 运行时间长：降低 `steps`、`n_traj`，或缩小绘图 `t_range`。

## 下一步

- 查阅并修改 `models/vdp_two_mode.py`，或编写提供 `build_sde(params)` 的自定义模型。
- 使用注册中心探索可用组件：
  - `integrator`：euler，milstein（当前为 euler 的别名）
  - `backend`：numpy
  - `visualizer`：phase_portrait，psd
