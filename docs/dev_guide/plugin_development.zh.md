---
description: 插件开发指南
---

# 插件开发指南

本指南概述了为 QPhase 框架开发扩展（插件）的过程。最常见的扩展点是 **模型**，它定义要仿真的物理系统。

## 插件契约

QPhase 使用**结构化子类型**（鸭子类型）。如果一个类满足相应协议（例如 `PluginBase`、`ModelBase`）定义的接口契约，则它被识别为有效插件。从框架基类继承是可选的，但不是必需的。

要实现插件，需要三个组件：
1.  **配置模式**：定义参数的 Pydantic 模型。
2.  **实现类**：包含逻辑的类。
3.  **注册**：插件注册表中的条目。

---

## 1. 定义配置模式

参数使用 **Pydantic** 模型定义。这确保了严格的类型验证和自动文档生成。

```python
from pydantic import BaseModel, Field

class MyModelConfig(BaseModel):
    """MyModel 的配置模式。"""

    # 必需参数（无默认值）
    chi: float = Field(..., description="Nonlinearity strength")

    # 带默认值的可选参数
    kappa: float = Field(1.0, gt=0, description="Decay rate (must be positive)")
```

---

## 2. 实现逻辑

实现类必须在其构造函数中接受配置对象和后端实例。

**关键要求**：所有数学操作必须使用注入的 `backend` 实例（按惯例为 `self.xp`）执行。直接使用 `numpy` 或 `torch` 会破坏硬件无关性。

### 示例：SDE 模型实现

SDE 模型通常实现 `drift` 和 `diffusion` 方法。

```python
from typing import Any, ClassVar
from qphase.backend.xputil import get_xp

class MyModel:
    # 注册表的元数据
    name: ClassVar[str] = "my_model"
    description: ClassVar[str] = "Kerr oscillator with additive noise"
    config_schema: ClassVar[type] = MyModelConfig

    def __init__(self, config: MyModelConfig, **kwargs: Any):
        self.cfg = config
        # 后端从 drift/diffusion 中的数据推断

    def drift(self, state: Any, t: float, params: dict) -> Any:
        """
        计算确定性漂移向量：A(X, t)
        dx = A(X, t)dt + B(X, t)dW
        """
        xp = get_xp(state)
        x = state
        chi = self.cfg.chi
        kappa = self.cfg.kappa

        # 使用 xp 进行张量操作
        # -1j * chi * |x|^2 * x - kappa * x
        term1 = -1j * chi * (xp.abs(x)**2) * x
        term2 = -kappa * x
        return term1 + term2

    def diffusion(self, state: Any, t: float, params: dict) -> Any:
        """
        计算扩散矩阵：B(X, t)
        """
        xp = get_xp(state)
        # 加性噪声：返回标量或常量张量
        return xp.sqrt(self.cfg.kappa)
```

### 示例：分析器实现

分析器处理原始仿真结果。

```python
from typing import Any, ClassVar
from qphase.backend.base import BackendBase
from qphase.core.protocols import ResultProtocol
from qphase_sde.result import SDEResult

class MyAnalyser:
    name: ClassVar[str] = "my_analyser"
    description: ClassVar[str] = "Calculates mean photon number"
    config_schema: ClassVar[type] = MyAnalyserConfig

    def __init__(self, config: MyAnalyserConfig, **kwargs: Any):
        self.cfg = config

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """
        处理仿真结果。
        """
        # 示例：计算轨迹的均值
        # data 预期是张量或 TrajectorySet
        if hasattr(data, "data"):
            traj = data.data
        else:
            traj = data

        mean_val = backend.mean(traj, axis=0)
        result_data = backend.abs(mean_val)**2

        return SDEResult(trajectory=result_data, kind="trajectory")
```

### 示例：带清单的引擎实现

引擎协调仿真。它们必须使用 `EngineManifest` 声明其依赖项。

```python
from typing import ClassVar
from qphase.core.protocols import EngineManifest

class MyEngine:
    name: ClassVar[str] = "my_engine"
    description: ClassVar[str] = "Custom simulation engine"
    config_schema: ClassVar[type] = MyEngineConfig

    # 声明依赖项
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"model", "backend"},
        optional_plugins={"analyser"}
    )

    def __init__(self, config: MyEngineConfig, plugins: dict):
        self.cfg = config
        self.model = plugins["model"]
        self.backend = plugins["backend"]
        # 处理可选插件
        self.analyser = plugins.get("analyser")

    def run(self):
        # ... 仿真循环 ...
        pass
```

---

## 3. 注册

插件可以通过两种机制注册：

### A. 本地注册（开发）
在项目根目录创建 `.qphase_plugins.yaml` 文件。这将插件命名空间和名称映射到 Python 类路径。

```yaml
model.my_model: "plugins.my_physics:MyModel"
analyser.my_analyser: "plugins.my_analysis:MyAnalyser"
```

### B. 包注册（分发）
如果将插件作为 Python 包分发，请在 `pyproject.toml` 中使用标准入口点。

```toml
[project.entry-points.qphase]
"model.my_model" = "my_package.models:MyModel"
"analyser.my_analyser" = "my_package.analysis:MyAnalyser"
```
