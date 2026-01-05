---
description: CLI 参考
---

# CLI 参考

`qphase` 命令行界面是与 QPhase 框架交互的主要工具。它用于项目初始化、任务执行、插件管理和配置生成。

## 全局选项

所有命令都支持以下全局标志：

*   `--help`：显示帮助信息并退出。
*   `--version`：显示已安装的版本号。

---

## 项目管理

### `qphase init`

在当前目录初始化一个新的 QPhase 项目。

```bash
qphase init
```

**功能：**

1.  创建标准目录结构：
    *   `configs/`：配置文件。
    *   `plugins/`：本地用户插件目录。
    *   `runs/`：仿真结果输出目录。
2.  生成默认的 `configs/global.yaml` 文件。

---

## 仿真执行

### `qphase run`

执行在 `configs/jobs/` 目录中定义的仿真任务。

```bash
qphase run [JOB_NAME] [OPTIONS]
```

*   **参数**：
    *   `JOB_NAME`：位于 `configs/jobs/` 中的任务配置文件名（不含扩展名）。
*   **选项**：
    *   `--list`：列出所有可用的任务配置并退出。
    *   `--verbose` / `-v`：启用详细调试日志。

**示例**：

```bash
# 执行单个任务
qphase run vdp_sde

# 列出可用任务
qphase run --list

# 使用详细日志运行
qphase run --verbose vdp_sde
```

---

## 插件管理

### `qphase list`

列出当前环境中所有已注册的插件。

```bash
qphase list [OPTIONS]
```

*   **选项**：
    *   `--category` / `-c`：按类别过滤插件（逗号分隔）。

**示例**：

```bash
qphase list
# 列出所有插件（backend, model, engine 等）

qphase list -c backend
# 仅列出 backend 插件
```

### `qphase show`

显示特定插件的详细信息，包括其描述、源代码位置和配置模式。

```bash
qphase show [PLUGIN_ID]... [OPTIONS]
```

*   **参数**：
    *   `PLUGIN_ID`：一个或多个 `namespace.name` 格式的插件标识符（例如 `model.vdp_two_mode`）。
*   **选项**：
    *   `--verbose` / `-v`：显示额外的元数据（例如文件路径、包版本）。

**示例**：

```bash
qphase show model.vdp_two_mode
qphase show backend.numpy --verbose
```

### `qphase template`

生成特定插件的配置模板。这对于复制粘贴到任务配置文件中非常有用。

```bash
qphase template [PLUGIN_ID]... [OPTIONS]
```

*   **参数**：
    *   `PLUGIN_ID`：一个或多个 `namespace.name` 格式的插件标识符。
*   **选项**：
    *   `--output` / `-o`：输出文件路径。默认为 `-`（标准输出）。
    *   `--format`：输出格式，`yaml`（默认）或 `json`。

**示例**：

```bash
# 将 YAML 模板打印到控制台
qphase template model.vdp_two_mode

# 保存到文件
qphase template model.vdp_two_mode -o my_config.yaml
```

---

## 配置管理

### `qphase config show`

显示当前配置。

```bash
qphase config show [OPTIONS]
```

*   **选项**：
    *   `--system` / `-s`：显示系统配置 (`system.yaml`) 而不是全局项目配置 (`global.yaml`)。

### `qphase config set`

设置 `global.yaml`（或 `system.yaml`）中的配置值。

```bash
qphase config set [KEY] [VALUE] [OPTIONS]
```

*   **参数**：
    *   `KEY`：点分隔的配置键（例如 `paths.output_dir`）。
    *   `VALUE`：要设置的值。
*   **选项**：
    *   `--system` / `-s`：更新系统配置而不是全局配置。

**示例**：

```bash
qphase config set paths.output_dir ./my_runs
```

### `qphase config reset`

将配置重置为默认值。

```bash
qphase config reset [OPTIONS]
```

*   **选项**：
    *   `--system` / `-s`：重置系统配置。
    *   `--force` / `-f`：强制重置而不确认。
