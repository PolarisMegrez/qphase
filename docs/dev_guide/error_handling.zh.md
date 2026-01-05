---
description: 错误处理策略
---

# 错误处理策略

QPhase 实现了一种结构化的错误处理策略，旨在为用户提供清晰、可操作的反馈，同时为调度器保持健壮的执行控制。

## 异常层次结构

所有框架特定的异常都继承自一个公共基类 `QPhaseError`。这允许顶层 CLI 包装器捕获已知错误并显示干净的消息，而不打印堆栈跟踪（除非启用详细模式）。

```python
class QPhaseError(Exception):
    """所有 QPhase 异常的基类。"""

class QPhaseConfigError(QPhaseError):
    """配置验证失败时引发。"""

class QPhasePluginError(QPhaseError):
    """插件加载或实例化失败时引发。"""

class QPhaseRuntimeError(QPhaseError):
    """仿真执行期间引发。"""

class QPhaseIOError(QPhaseError):
    """文件输入/输出操作期间引发。"""
```

## 错误传播

1.  **验证阶段**：配置加载期间的错误（例如缺少字段、无效类型）会被提前捕获并作为 `QPhaseConfigError` 引发。CLI 显示来自 Pydantic 的具体验证消息。
2.  **依赖检查**：如果任务缺少必需的插件（如 `EngineManifest` 中定义的），则在执行开始之前引发 `QPhaseConfigError`。
3.  **执行阶段**：任务中发生的异常（例如数值不稳定、运行时断言）由 `Scheduler` 捕获。
    *   记录异常。
    *   在 `JobResult` 中将任务标记为 `failed`。
    *   调度器继续执行下一个独立任务（除非启用 `fail_fast`）。

## 日志记录

QPhase 使用标准 Python `logging` 模块。
*   **控制台**：默认情况下，只显示 `INFO` 级别及以上。
*   **文件**：如果配置，所有日志（包括 `DEBUG`）都会写入日志文件。
*   **格式**：日志包含时间戳和模块名称，以帮助调试。
