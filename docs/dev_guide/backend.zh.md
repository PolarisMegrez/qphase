---
description: 后端系统
---

# 后端系统

**后端系统**是 QPhase 的计算抽象层。它通过提供张量操作的统一接口来解决硬件异构性问题，使仿真代码能够与底层计算库（如 NumPy、PyTorch、CuPy）保持无关。

## 架构目标

主要目标是将**物理模型**与**计算实现**解耦。这允许单个模型实现能够：
1.  在 CPU 上执行，用于调试或小规模仿真（通过 NumPy）。
2.  在 GPU 上执行，用于高性能并行仿真（通过 PyTorch 或 CuPy）。
3.  支持自动微分（通过 PyTorch/JAX），无需修改代码。

## 后端协议

该系统的核心是 `BackendBase` 协议，它定义了所有计算后端的契约。它标准化了数组创建、线性代数、随机数生成和常用实用函数的 API。

**严格的硬件无关性**：该协议旨在防止在仿真循环期间发生"主机传输"（在 CPU 和 GPU 之间移动数据）。所有操作（包括重塑和索引）必须通过后端接口执行。

```python
@runtime_checkable
class BackendBase(Protocol):
    """张量操作的抽象接口。"""

    # 标识
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...

    # 数组创建
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def arange(self, start: int, stop: int | None = None, step: int = 1, dtype: Any | None = None) -> Any: ...

    # 形状操作
    def expand_dims(self, x: Any, axis: int) -> Any: ...
    def repeat(self, x: Any, repeats: int, axis: int | None = None) -> Any: ...
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any: ...
    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any: ...

    # 数学与线性代数
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...
    def isnan(self, x: Any) -> Any: ...

    @property
    def pi(self) -> float: ...

    # 随机数生成
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
```

## 实现策略

### 包装器模式
具体后端（例如 `NumpyBackend`、`TorchBackend`）通过包装相应的库调用来实现 `BackendBase` 协议。这确保了方法签名（参数、返回类型）在所有实现中保持一致，消除了库之间的 API 差异（例如 `np.concatenate` vs `torch.cat`，`np.repeat` vs `torch.repeat_interleave`）。

### 张量调度
在仿真内核中，后端实例通常作为 `self.xp` 注入（遵循数组 API 标准约定）。所有数学操作都通过此实例调度。

**示例：**
```python
# 硬件无关的实现
def drift(self, state):
    # self.xp 可以是 numpy 或 torch
    # 使用 self.xp.pi 确保在需要时使用正确的标量类型
    phase = 2.0 * self.xp.pi * state
    return -1j * self.xp.einsum("ij,j->i", self.hamiltonian, phase)
```

## 可用后端

### 1. NumPy 后端 (`backend: numpy`)
*   **目标**：CPU。
*   **用例**：开发、调试、小规模仿真。
*   **特征**：默认双精度、确定性执行、广泛兼容。

### 2. PyTorch 后端 (`backend: torch`)
*   **目标**：CPU / GPU (CUDA/MPS)。
*   **用例**：大规模并行仿真、基于梯度的优化。
*   **特征**：支持 `float32`/`float64`、自动设备管理、批量操作。使用 `torch.repeat_interleave` 实现 `repeat`。

### 3. CuPy 后端 (`backend: cupy`)
*   **目标**：NVIDIA GPU。
*   **用例**：不需要 PyTorch 开销的高性能计算。
*   **特征**：GPU 上的 NumPy 兼容 API。

## 扩展后端

开发者可以通过实现满足 `BackendBase` 协议的类并在 `backend` 命名空间下注册，来引入对新库（例如 JAX、TensorFlow）的支持。
