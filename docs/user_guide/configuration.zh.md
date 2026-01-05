---
description: 任务配置指南
---

# 任务配置

QPhase 采用分层式的 YAML 配置系统，旨在确保仿真的可重复性、灵活性及易用性。

## 编写配置文件

QPhase 中的“任务 (Job)”由位于 `configs/jobs/` 目录下的 YAML 文件定义。每个文件代表一个仿真任务（若启用了参数扫描，则代表一组任务）。

### 使用模板

创建新配置最便捷的方式是使用 CLI 生成模板。

```bash
# 为特定模型生成模板
qphase template model.vdp_two_mode > configs/jobs/my_new_job.yaml
```

生成后，可根据具体需求编辑该文件。

### 实例化插件

QPhase 基于插件架构。在配置文件中定义某个部分（如 `model` 或 `backend`）即指示 QPhase 实例化相应的插件类。

例如：

```yaml
model:
  vdp_two_mode:   # 对应插件 ID "model.vdp_two_mode"
    D: 0.5        # 这些参数将传递给插件的 __init__ 方法
```

### 配置层级

为了确定仿真任务的最终设置，QPhase 会合并来自多个源的配置。优先级从高到低如下：

1.  **任务配置** (`configs/jobs/*.yaml`)：针对单次仿真运行的特定设置。此处的设置将覆盖所有其他配置。
2.  **全局配置** (`configs/global.yaml`)：项目级的默认设置（例如默认后端、日志偏好）。
3.  **系统默认值**：QPhase 包及其插件提供的内置默认值。

## 任务配置详解

任务配置文件定义了仿真的具体参数。

### 通用字段

| 字段 | 类型 | 描述 |
| :--- | :--- | :--- |
| `name` | `str` | **必填**。任务的唯一标识符，用于日志记录及输出文件名。 |
| `engine` | `dict` | **必填**。仿真引擎的配置。必须包含且仅包含一个与引擎名称对应的键（例如 `sde`）。 |
| `input` | `str` | **可选**。指定输入源，例如上游任务的名称。 |
| `output` | `str` | **可选**。指定输出目标。 |

### QPhase-SDE 字段

当使用 SDE 引擎 (`qphase-sde`) 时，可使用以下顶级键。这些键对应于 SDE 求解器使用的插件类型。

| 字段 | 描述 |
| :--- | :--- |
| `model` | 物理模型插件的配置（例如 `vdp_two_mode`, `kerr_cavity`）。定义漂移项和扩散项。 |
| `backend` | 计算后端插件的配置（例如 `numpy`, `torch`）。定义数组的处理方式。 |
| `integrator` | SDE 积分器插件的配置（例如 `euler`, `milstein`）。定义数值步进方案。 |

### QPhase-Viz 字段

当使用可视化引擎 (`qphase-viz`) 时，相关字段如下：

| 字段 | 描述 |
| :--- | :--- |
| `analyser` | 数据分析插件的配置（例如 `psd`, `trajectory`）。用于处理原始仿真数据。 |
| `visualizer` | 绘图插件的配置。定义分析数据的渲染方式。 |

## 参数扫描

QPhase 内置了对参数扫描的支持，允许通过单个配置文件运行具有不同参数的多个仿真。

### 笛卡尔积 (网格搜索)

若为多个参数提供了列表，QPhase 将生成所有可能组合的任务（笛卡尔积）。

```yaml
model:
  kerr_cavity:
    chi: [1.0, 2.0]       # 2 个值
    epsilon: [0.1, 0.5, 1.0] # 3 个值
# 总任务数 = 2 * 3 = 6
```

### 联动扫描 (Zipped Scanning)

若需同步扫描参数（例如 `chi` 和 `epsilon` 一一对应变化），可使用联动扫描方法。这需要在 `system.yaml` 中进行配置（见下文），或使用插件支持的特定语法。

*（注：当前默认行为为笛卡尔扫描。联动扫描需在 `system.yaml` 中启用 `parameter_scan.method = "zip"`）*

## 系统配置

`configs/system.yaml` 文件控制 QPhase 框架本身的行为，而非具体的仿真参数。

### 关键设置

*   **`paths`**:
    *   `output_dir`: 所有仿真输出的根目录（默认：`runs/`）。
*   **`logging`**:
    *   `level`: 全局日志级别 (INFO, DEBUG, WARNING)。
*   **`parameter_scan`**:
    *   `enabled`: 启用/禁用参数扫描（默认：`true`）。
    *   `method`: 扫描方法 (`cartesian` 或 `zip`)。
    *   `numbered_outputs`: 是否为扫描任务的输出目录添加编号后缀（默认：`true`）。
