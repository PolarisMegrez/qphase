---
description: 协议系统
---

# 协议系统

QPhase 使用 **Python 协议**（在 PEP 544 中引入）来定义核心框架与其可扩展组件之间的接口。这种设计选择倾向于**结构化子类型**（鸭子类型），而非传统的由抽象基类（ABC）强制执行的名义子类型。

## 结构化子类型 vs 名义子类型

### 名义子类型（ABC）
在名义系统中，一个类只有在显式继承另一个类时才是其子类型。
*   **要求**：`class MyPlugin(PluginBase): ...`
*   **缺点**：这创建了对框架代码的硬依赖。第三方插件必须导入基类，导致潜在的版本冲突和更紧密的耦合。

### 结构化子类型（协议）
在结构化系统中，如果一个类实现了所需的方法和属性，则它是子类型，无论继承关系如何。
*   **要求**：类只需要具有正确的方法。
*   **优点**：解耦。插件可以在不导入 `qphase` 的情况下开发、测试和分发。依赖仅在于*接口契约*，而非实现。

## 核心协议

框架定义了插件必须满足的几个关键协议。

### 1. `PluginBase`
所有可发现组件的基本契约。

```python
@runtime_checkable
class PluginBase(Protocol):
    """任何 QPhase 插件的最小契约。"""

    # 元数据（类变量）
    name: ClassVar[str]                 # 唯一标识符
    description: ClassVar[str]          # 人类可读的描述
    config_schema: ClassVar[type[Any]]  # 配置的 Pydantic 模型

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        """
        初始化插件。

        参数：
            config：已验证的配置对象（config_schema 的实例）。
            **kwargs：由注册表注入的额外依赖项。
        """
        ...
```

### 2. `EngineBase` 和 `EngineManifest`
仿真引擎的契约。引擎必须通过清单声明其依赖项。

```python
@dataclass
class EngineManifest:
    """
    声明引擎的依赖项。
    """
    required_plugins: set[str] = field(default_factory=set)
    optional_plugins: set[str] = field(default_factory=set)
    defaults: dict[str, str] = field(default_factory=dict)

@runtime_checkable
class EngineBase(PluginBase, Protocol):
    # 声明依赖项的清单
    manifest: ClassVar[EngineManifest] = EngineManifest()

    def run(
        self,
        data: Any | None = None,
        *,
        progress_cb: Callable[[float | None, float | None, str, str | None], None]
        | None = None,
    ) -> ResultProtocol:
        """执行仿真并返回结果对象。"""
        ...
```

### 3. `BackendBase`
计算后端的契约（参见[后端系统](backend.md)）。

## 运行时验证

虽然协议主要是静态分析工具（由 MyPy/Pyright 使用），但 QPhase 利用 `@runtime_checkable` 装饰器在插件实例化期间执行运行时验证。这允许注册表在加载的插件未能满足所需契约时引发信息性错误。
