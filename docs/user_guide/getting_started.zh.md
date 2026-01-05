---
description: QPhase 入门指南
---

# 入门指南

本指南将介绍 QPhase 的安装流程，并演示如何运行第一个仿真任务。

## 1. 安装

### 前置要求

*   **Python**：版本 3.10 或更高。
*   **操作系统**：Windows、macOS 或 Linux。

### 推荐：虚拟环境

强烈建议使用虚拟环境，以避免与其他 Python 包发生冲突。

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 从 PyPI 安装（推荐）

QPhase 已发布至 PyPI，可直接通过 pip 安装：

```bash
pip install qphase
```

此命令仅安装核心框架。**为了运行本指南中的示例，还需要安装 SDE（随机微分方程）扩展包：**

```bash
pip install qphase-sde
```

### 从源码安装（面向开发者）

若需参与 QPhase 开发或使用最新的未发布功能，请执行以下步骤：

```bash
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
pip install -e packages/qphase[standard]
pip install -e packages/qphase_sde
pip install -e packages/qphase_viz
```

## 2. 初始化项目

建议为仿真项目创建独立的文件夹，以便于组织配置文件与结果数据。

```bash
# 创建项目文件夹
mkdir my_research
cd my_research

# 初始化 QPhase 项目结构
qphase init
```

该命令将创建以下目录结构：

*   `configs/`：**配置文件目录**。
    *   包含定义仿真任务 (Jobs) 和全局设置的 YAML 配置文件。
    *   详见 [任务配置](configuration.md)。
*   `plugins/`：**自定义代码目录**。
    *   用于存放用户自定义的 Python 模块（如模型、后端等），QPhase 将自动加载该目录下的插件。
    *   详见 [插件开发](../dev_guide/plugin_development.md)。
*   `runs/`：**仿真数据目录**。
    *   所有的仿真输出、日志及复现快照均按日期和运行 ID 组织在此目录下。
    *   详见 [结果与复现](output.md)。

## 3. 创建第一个任务

“任务 (Job)” 是指由 YAML 文件定义的单次仿真运行。
请新建文件 `configs/jobs/test_run.yaml` 并粘贴以下内容：

```yaml
# configs/jobs/test_run.yaml
name: test_run

# 1. 选择引擎 (SDE 求解器)
engine:
  sde:
    t_end: 10.0
    dt: 0.01
    n_traj: 100

# 2. 选择物理模型
# (此处使用内置的示例模型)
plugins:
  model:
    vdp_two_mode:  # 内置的范德波尔振荡器
      D: 1.0       # 扩散强度

  backend:
    numpy:         # 在 CPU 上运行
      float_dtype: float64
```

## 4. 运行仿真

使用 `qphase run` 命令执行任务：

```bash
qphase run test_run
```

若一切正常，终端将显示进度条，并在完成后提示结果保存路径（通常位于 `runs/` 目录下）。
