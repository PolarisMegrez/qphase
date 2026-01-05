---
description: CLI 架构
---

# CLI 架构

**命令行界面 (CLI)** 是用户交互的主要入口点。它基于 **Typer** 库构建，该库利用 Python 类型提示自动生成命令行解析器和帮助文档。

## 命令结构

CLI 以层次化命令组结构组织，根节点为 `qphase` 入口点。

*   `qphase`（根）
    *   `init`：项目初始化。
    *   `run`：仿真执行。
    *   `list`：插件发现和列表。
    *   `show`：插件内省。
    *   `template`：配置脚手架。

## 实现细节

### 入口点

主应用程序入口点定义在 `qphase.main:app`。这个 `Typer` 实例聚合子命令并处理全局标志（例如 `--verbose`、`--version`）。

### 命令注册

命令使用 `@app.command()` 装饰器注册。Typer 检查函数签名以确定参数类型和选项。

```python
@app.command()
def jobs(
    job_name: str = typer.Argument(..., help="Job name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """执行仿真任务。"""
    # ... 实现 ...
```

### 可扩展性

虽然核心命令是硬编码的，但 CLI 架构允许未来的可扩展性。注册表系统包含一个 `command` 命名空间，用于从插件动态加载额外的 CLI 子命令。这将允许第三方包使用自定义功能扩展 `qphase` 工具（例如 `qphase plot`、`qphase analyze`）。

## 与调度器的集成

CLI 充当 `Scheduler` 的轻量客户端。当调用 `qphase run` 时：
1.  解析命令行参数。
2.  加载 `SystemConfig`。
3.  实例化 `Scheduler`。
4.  将执行委托给 `scheduler.run()`。

这种分离确保核心执行逻辑不与 CLI 界面耦合，允许在需要时从程序化方式（例如从 Jupyter notebook）触发仿真。
