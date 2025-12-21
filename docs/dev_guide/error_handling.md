---
layout: default
title: 9 错误处理与日志
---

# 9 错误处理与日志

核心包定义了统一的异常层次和日志系统，用于提供清晰的错误分类、可追踪的错误链和灵活的日志配置。

### 9.1 异常层次设计

**异常树结构**：

```
QPhaseError (基类)
├── QPhaseIOError          # 文件/网络 I/O 错误
├── QPhaseConfigError      # 配置验证错误
├── QPhasePluginError      # 插件发现/实例化错误
├── QPhaseSchedulerError   # 任务调度错误
├── QPhaseRuntimeError     # 引擎执行错误
└── QPhaseCLIError         # CLI 参数/执行错误

QPhaseWarning (警告基类)
```

**设计原则**：

1. **单一根异常**：所有框架异常继承自 QPhaseError，便于统一捕获
2. **按职责分类**：每类异常对应特定的错误源
3. **异常链保留**：使用 `raise ... from e` 保留原始异常栈
4. **警告分离**：QPhaseWarning 独立于异常体系，用于非致命问题

### 9.2 各异常类型的使用场景

**QPhaseIOError**：

用于文件系统和网络 I/O 操作失败：

```python
# 文件不存在
if not path.exists():
    raise QPhaseIOError(f"File not found: {path}")

# 写入失败
try:
    save_global_config(config, path)
except Exception as e:
    raise QPhaseIOError(f"Failed to save global config to {path}: {e}") from e
```

典型触发场景：
- 配置文件不存在
- 输出目录无写入权限
- 快照保存失败

**QPhaseConfigError**：

用于配置验证和解析错误：

```python
# YAML 解析失败
try:
    data = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise QPhaseConfigError(f"Failed to parse YAML file {path}: {e}") from e

# Pydantic 验证失败
try:
    validated = schema.model_validate(params)
except ValidationError as e:
    raise QPhaseConfigError(f"Invalid configuration for '{plugin_type}:{name}': {e}") from e

# 缺少必需字段
if not name:
    raise QPhaseConfigError(f"Plugin config for '{plugin_type}' missing 'name'")
```

典型触发场景：
- YAML 语法错误
- 缺少必需配置字段
- 配置值类型错误
- Pydantic 验证失败

**QPhasePluginError**：

用于插件发现、导入和实例化错误：

```python
# 插件未注册
if entry is None:
    raise QPhasePluginError(f"Plugin '{nm}' not found in namespace '{ns}'")

# 导入失败
try:
    obj = self._import_target(entry.target)
except Exception as e:
    raise QPhasePluginError(
        f"Failed to import plugin '{nm}' from '{entry.target}': {e}"
    ) from e

# 实例化失败
try:
    instance = registry.create_plugin_instance(plugin_type, config_data)
except Exception as e:
    raise QPhasePluginError(f"Failed to create plugin '{plugin_type}': {e}") from e
```

典型触发场景：
- 请求未注册的插件
- 插件模块导入失败（依赖缺失）
- 插件构造函数抛出异常
- 输入加载器执行失败

**QPhaseSchedulerError**：

用于任务调度层的错误（预留，当前版本较少使用）：

```python
# 任务依赖解析失败
raise QPhaseSchedulerError(f"Circular dependency detected in job '{job.name}'")

# 资源分配失败
raise QPhaseSchedulerError(f"Cannot allocate GPU for job '{job.name}'")
```

**QPhaseRuntimeError**：

用于引擎执行时的错误，包装来自 Engine.run() 的异常：

```python
try:
    output = engine.run(data=input_data)
except Exception as e:
    log.error(f"Job execution failed: {e}")
    raise QPhaseRuntimeError(
        f"Job '{job.name}' execution failed in engine '{job.get_engine_name()}': {e}"
    ) from e
```

典型触发场景：
- SDE 积分发散
- 模型计算异常
- 后端操作失败

**QPhaseCLIError**：

用于命令行参数和执行错误：

```python
# 参数验证失败
raise QPhaseCLIError("Invalid command arguments")

# 命令执行失败
raise QPhaseCLIError(f"Command '{cmd}' failed with exit code {code}")
```

### 9.3 异常链与错误上下文

**异常链模式**：

核心包使用 `raise ... from e` 语法保留原始异常：

```python
try:
    config = schema.model_validate(raw_data)
except ValidationError as e:
    raise QPhaseConfigError(
        f"Invalid configuration for plugin '{name}': {e}"
    ) from e  # 保留原始 ValidationError
```

这样做的好处：
1. 用户看到高层错误信息（QPhaseConfigError）
2. 开发者通过 `__cause__` 访问原始异常
3. 完整的异常栈在 traceback 中可见

**错误消息格式**：

```
[层级标识] 上下文信息: 具体错误
```

示例：
```
QPhasePluginError: Failed to import plugin 'cupy' from 'qphase.backend.cupy_backend:CuPyBackend': No module named 'cupy'
```

### 9.4 日志系统

**单例 Logger**：

核心包使用名为 `"qphase"` 的共享 Logger：

```python
_logger: logging.Logger | None = None

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("qphase")
        _logger.setLevel(logging.INFO)
        # 添加默认控制台处理器
        if not _logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            h.setFormatter(fmt)
            _logger.addHandler(h)
    return _logger
```

**设计选择**：
- 延迟初始化：首次调用时创建
- 单例模式：全局共享同一 Logger
- 默认 INFO 级别：平衡信息量和性能

### 9.5 日志配置 API

**configure_logging 函数**：

```python
def configure_logging(
    verbose: bool = False,
    log_file: str | None = None,
    as_json: bool = False,
    suppress_warnings: bool = False,
) -> None:
```

**参数说明**：

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `verbose` | bool | False | True 时设为 DEBUG 级别 |
| `log_file` | str\|None | None | 日志文件路径（追加模式） |
| `as_json` | bool | False | 输出 JSON 行格式 |
| `suppress_warnings` | bool | False | 将警告提升为 ERROR 级别 |

**配置流程**：

1. 清空现有处理器（避免重复）
2. 设置日志级别（DEBUG 或 INFO）
3. 创建控制台处理器
4. 选择格式化器（文本或 JSON）
5. 可选添加文件处理器
6. 配置警告捕获

**输出格式**：

文本格式：
```
2025-01-15 10:30:45,123 [INFO] qphase: Starting job execution
2025-01-15 10:30:45,456 [ERROR] qphase: Job execution failed: ...
```

JSON 格式：
```json
{"time":"2025-01-15 10:30:45,123","level":"INFO","logger":"qphase","msg":"Starting job execution"}
```

**警告集成**：

通过 `logging.captureWarnings(True)` 将 Python warnings 路由到日志系统：

```python
if suppress_warnings:
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)
else:
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
```

### 9.6 废弃标记装饰器

**deprecated 装饰器**：

用于标记即将移除的 API：

```python
@deprecated("Use new_api() instead")
def old_api():
    return 42
```

**行为**：
1. 首次调用时发出 QPhaseWarning（代码 [990]）
2. 同时记录到 Logger
3. 后续调用不再重复警告
4. 不影响原函数执行

**输出示例**：
```
[990] DEPRECATED: old_api: Use new_api() instead
```

### 9.7 错误处理最佳实践

**对于插件开发者**：

1. **不要吞掉异常**：让错误传播到调度层
2. **添加上下文**：重新抛出时包含有用信息
3. **使用正确的异常类型**：配置问题用 QPhaseConfigError
4. **记录日志**：在抛出前记录 DEBUG 级别信息

```python
def my_plugin_method(self):
    log = get_logger()
    try:
        result = risky_operation()
    except SomeError as e:
        log.debug(f"Operation failed with details: {e}")
        raise QPhaseRuntimeError(f"MyPlugin operation failed: {e}") from e
```

**对于 CLI 开发者**：

1. **捕获 QPhaseError**：统一处理框架异常
2. **设置退出码**：错误时返回非零退出码
3. **用户友好输出**：使用 Rich 格式化错误消息

```python
try:
    scheduler.run(job_list)
except QPhaseError as e:
    log.error(str(e))
    raise typer.Exit(code=1) from e
except Exception as e:
    log.error(f"Unexpected error: {e}")
    raise typer.Exit(code=1) from e
```

**对于调试**：

1. 使用 `--verbose` 启用 DEBUG 日志
2. 使用 `--log-file` 保存完整日志
3. 检查异常的 `__cause__` 属性获取原始错误
4. JSON 格式日志便于程序化分析
