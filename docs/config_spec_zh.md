# QPhaseSDE v0.1.3 配置说明（Triad YAML）

本文档描述 v0.1.3 版本中 `qps run sde` 所需的 YAML 配置结构。配置分为三个部分：`model`、`profile`、`run`。

## 1. model

定义物理模型与初始化。

必填字段：
- module：字符串——Python 模块路径或 .py 文件路径（需提供 `build_sde(params)`）
- function：字符串——模块中调用的函数（如 `build_sde`）
- params：对象——传入 `build_sde` 的参数映射
- ic：列表——复数模式幅度的初始条件
  - 可接受一维列表（单个初值向量）或二维列表（多个初值向量）
  - 每个元素必须是可被 Python `complex()` 解析的字符串，如 `"7.0+0.0j"`
  - 形状：
    - 一维：[`"7.0+0.0j"`, `"0.0-7.0j"`] → 单个长度为 `n_modes` 的向量
    - 二维：[[...], [...], ...] → 要么长度为 1（广播），要么长度为 `n_traj`
- noise：对象
  - kind：`independent` | `correlated`
  - covariance：当 `kind=correlated` 时需要（实对称，半正定）

示例：
```yaml
model:
  module: models.vdp_two_mode
  function: build_sde
  params:
    omega_a: 0.005
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.01
    g: 0.5
    D: 1.0
  ic: ["7.0+0.0j", "0.0-7.0j"]
  noise:
    kind: independent
```

说明：
- CLI 会校验每个 IC 向量的长度与 `model.n_modes` 一致。
- 单个 IC 向量会广播到所有轨迹；否则需提供 `n_traj` 个向量。

## 2. profile（B 类，含默认值）

控制与物理模型无关的执行配置。

字段：
- backend：`numpy`（默认）| `numba`（预留）
- solver：`euler`（默认）| `milstein`（v0.1.3 中占位，回退至 euler）
- save：
  - root：字符串（默认 `runs`）——输出根目录
  - save_every：整数（可选）——时序数据保存抽样间隔
  - save_timeseries：布尔（必填）——是否保存每个 IC 的时序 NPZ
  - save_psd_complex：布尔（必填）——是否计算/保存复信号 PSD NPZ
  - save_psd_modular：布尔（必填）——是否计算/保存模值 PSD NPZ
- visualizer（可选）：matplotlib 参数与 PSD 规范
  - phase_portrait：
    - Re_Im：`Re-Im` 相图的绘图参数（推荐；接受 `re_im`）
    - abs_abs：`|.|-|.|` 相图的绘图参数
  - psd：
    - convention：`symmetric`（等价 `unitary`）或 `pragmatic`
    - x_scale：`linear` 或 `log`
    - y_scale：`linear` 或 `log`

示例：
```yaml
profile:
  backend: numpy
  solver: euler
  save:
    root: runs
    save_every: 20
    save_timeseries: true
    save_psd_complex: true
    save_psd_modular: false
  visualizer:
    phase_portrait:
      re_im:
        linewidth: 0.8
        alpha: 0.6
      abs_abs:
        linewidth: 0.8
        alpha: 0.6
    psd:
      convention: symmetric
      x_scale: linear
      y_scale: log
```

## 3. run（C 类，按需求必填）

定义数值积分与请求的可视化。

字段：
- time：
  - dt：浮点（必填）
  - steps：整数（必填）
  - t0：浮点（默认 0.0）
- trajectories：
  - n_traj：整数（必填）
  - master_seed：整数（可选）——主随机种子（用于派生每条轨迹的种子），或
  - seed_file：字符串（可选）——包含 N 个种子的文件路径（每轨迹一个）
  - rng_stream：`per_trajectory` | `batched`（可选；默认 `per_trajectory`）——控制随机数流策略：
    - `per_trajectory`：每条轨迹独立 RNG 流（当 n_traj 改变时序仍稳定）
    - `batched`：单一 RNG 流进行向量化采样（更快；当 n_traj/顺序改变时序会改变）
- visualizer（可选）：
  - phase_portrait：图列表（每项对应一张图）
    - kind：`Re_Im`（推荐）或 `re_im`，或 `abs_abs`（必填）
    - modes：整数列表——`re_im` 需 1 个索引；`abs_abs` 需 2 个索引（必填）
    - t_range：[t_start, t_end]（可选）——绘图使用的时间段
  - psd：PSD 图列表
    - kind：`complex` | `modular`（必填）
    - modes：整数列表（一个或多个，必填）
    - xlim：[xmin, xmax]（可选）
    - t_range：[t_start, t_end]（可选）

示例：
```yaml
run:
  time:
    dt: 0.001
    steps: 200
  trajectories:
    n_traj: 4
    master_seed: 42
  visualizer：
    phase_portrait：
      - kind: Re_Im
        modes: [0]
      - kind: abs_abs
        modes: [0, 1]
        t_range: [0.05, 0.15]
    psd:
      - kind: complex
        modes: [0, 1]
        xlim: [-0.5, 0.5]
        t_range: [20.0, 100.0]
```

## 校验摘要

- 模式确保必要字段存在。
- `model.ic` 规范化为二维列表，并校验其可解析为复数。
- `run.viz.phase` 校验 `kind/modes` 的形状与 `t_range` 合法性。
- 运行时，CLI 会校验 IC 向量长度是否等于 `model.n_modes`，以及 IC 列表长度为 1 或 `n_traj`。

## 运行时行为

- 运行输出：
  - `time_series/timeseries_icXX.npz`（每个 IC 一份；包含 data, t0, dt）
  - `config_snapshot/config.json` 与 `config_snapshot/triad.yaml`
  - 当请求绘图时，每个 IC 的图片保存至 `figures/icXX/*.png`
  - 当保存开关开启时，PSD 保存至 `psd/icXX/psd_{kind}_{convention}.npz`
- 重绘：
  - `qps analyze phase --from-run <run_dir>` 使用保存的绘图配置与样式。
  - `qps analyze psd --from-run <run_dir>` 使用保存的 PSD 配置与样式。
  - 或使用 `--specs-json` 临时覆盖图形规格，无需重新计算。

## 版本说明与容量保护

- v0.1.3 支持 NumPy + Euler–Maruyama，Milstein 为占位实现；支持多 IC 语义：
  - 当 `model.ic` 包含多个向量时，视为多个独立配置（除 IC 不同外皆相同）逐一仿真。
  - 每个 IC 的时序保存为 `time_series/timeseries_icXX.npz`。
  - 每个 IC 的图片保存到 `figures/icXX/`。
  - 当开关开启时保存 PSD。
- 为避免磁盘占用过大，CLI 在保存时序前会估算容量；若预计超过 1 GiB，默认拒绝运行。可通过 `--max-storage-gb` 覆盖限制。
- 后续版本将扩展求解器、后端与可视化类型。
