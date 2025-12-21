---
layout: default
title: Backend System
parent: Developer Guide
nav_order: 3
---

# Backend System

The **Backend System** is the computational abstraction layer of QPhase. It defines a unified interface via the `BackendBase` Protocol, solving the problem of multi-backend compatibility (CPU/GPU, NumPy/Torch).

## Design Goals

*   **Decoupling**: Algorithm code depends only on the Backend interface, not on specific libraries.
*   **Seamless Switching**: Switch between NumPy, PyTorch, or CuPy via configuration without modifying algorithm code.
*   **Performance**: Choose the right backend for the job (CPU vs. GPU, JIT vs. Interpreted).
*   **Dependency Isolation**: Algorithm code does not directly import heavy dependencies like PyTorch.

## The Backend Protocol

The `BackendBase` protocol defines the minimal set of operations required for scientific computing.

```python
@runtime_checkable
class BackendBase(Protocol):
    """Minimal backend protocol for array ops, linalg, RNG, and helpers."""

    # Identification
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...
    def capabilities(self) -> dict[str, Any]: ...

    # Array Creation & Conversion
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty_like(self, x: Any) -> Any: ...
    def copy(self, x: Any) -> Any: ...

    # Math Operations
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...

    # Random Number Generation
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]: ...

    # Complex Number Support
    def real(self, x: Any) -> Any: ...
    def imag(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any: ...

    # FFT
    def fft(self, x: Any, axis: int = -1, norm: Literal["backward", "ortho", "forward"] | None = None) -> Any: ...
    def fftfreq(self, n: int, d: float = 1.0) -> Any: ...
```

### Design Principles

*   **Minimal Interface**: Defines only ~20 essential methods covering 80% of common computational scenarios.
*   **Domain Agnostic**: Contains no SDE-specific concepts; applicable to general scientific computing.
*   **Type Consistency**: All methods accept and return the same abstract type (`Any`), avoiding generic constraints that complicate implementation.

## Core Interface Categories

### 1. Identification & Metadata
*   `backend_name()`: Returns the backend identifier (e.g., "numpy", "torch").
*   `device()`: Returns the device string (e.g., "cpu", "cuda:0").
*   `capabilities()`: Reports supported features and environment flags.

### 2. Array Creation
*   `asarray()`: Zero-copy conversion if possible (Recommended).
*   `array()`: Always creates a copy.
*   `empty_like()`: Creates an uninitialized array with the same shape and dtype.

### 3. Math Operations
*   `einsum()`: Einstein summation convention. Highly compact and efficient for tensor operations.
    *   Example: `einsum("ijk,kl->ijl", A, B)` for tensor contraction.

### 4. Random Number Generation (RNG)
*   **Handle-based**: `rng()` returns a passable RNG object, not a pure function.
*   **Deterministic**: Same seed produces the same sequence.
*   **Independent Streams**: `spawn_rngs()` generates multiple independent RNGs for parallel trajectories.

### 5. Complex Numbers
*   `real()` / `imag()`: Returns views (zero-copy) where possible.

## Implementations

### NumPy Backend (Reference)
The baseline CPU implementation. All other backends should match its behavior.

*   **Optimization**: Enables `optimize=True` for `einsum` path optimization.
*   **RNG**: Uses `numpy.random.SeedSequence` for robust seeding.

### PyTorch Backend
Supports dynamic device selection (CPU/GPU) and automatic type mapping.

*   **Type Mapping**: Automatically maps Python `float` to `torch.float64` and `complex` to `torch.complex128`.
*   **Device Awareness**: `zeros` and `asarray` accept a device context.
*   **RNG Adapter**: Wraps `torch.Generator` to provide a unified interface compatible with NumPy's seeding logic.

## Capability Negotiation

Backends report their capabilities via a dictionary:

```python
def capabilities(self) -> dict:
    return {
        "device": self.device(),
        "optimized_contractions": True,  # Is einsum optimized?
        "supports_complex_view": False,  # Do real/imag return views?
        "real_imag_split": True,
        "stack": True,
        "to_device": True,
        "numpy": True,  # Is underlying implementation NumPy-based?
    }
```

Callers should check `capabilities()` rather than assuming all backends support all features.
        "device": self.device(),
        "optimized_contractions": True,   # cuDNN 优化
        "supports_complex_view": True,    # PyTorch 原生支持复数视图
        "real_imag_split": True,
        "stack": True,
        "to_device": True,                # 支持设备迁移
        "torch": True,
    }
```

### 6.4 Numba Backend - JIT 加速与专用内核

**定位**：CPU 后端的性能优化版本，针对常见 contraction 模式提供 JIT 加速。

**JIT 编译内核**：
```python
@njit(cache=True, fastmath=False)
def _einsum_tnm_tm_to_tn(L: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """编译的 einsum 内核：(tnm, tm) -> (tn)"""
    T, N, M = L.shape
    out = np.empty((T, N), dtype=np.complex128)
    for t in range(T):
        for n in range(N):
            acc_r = 0.0
            acc_i = 0.0
            for m in range(M):
                c = L[t, n, m]
                w = dW[t, m]
                acc_r += c.real * w
                acc_i += c.imag * w
            out[t, n] = acc_r + 1j * acc_i
    return out
```

**内核选择逻辑**：
```python
def einsum(self, subscripts: str, *operands: Any) -> Any:
    # 专用优化路径：匹配特定模式
    if subscripts == "tnm,tm->tn" and len(operands) == 2:
        L = operands[0]
        dW = operands[1]
        L_arr = np.asarray(L, dtype=np.complex128)
        dW_arr = np.asarray(dW, dtype=np.float64)
        return _einsum_tnm_tm_to_tn(L_arr, dW_arr)

    if subscripts == "tm,mk->tk" and len(operands) == 2:
        z = operands[0]
        chol_T = operands[1]
        z_arr = np.asarray(z, dtype=np.float64)
        cholT_arr = np.asarray(chol_T, dtype=np.float64)
        return _einsum_tm_mk_to_tk(z_arr, cholT_arr)

    # 回退到 NumPy（一般情况）
    return np.einsum(subscripts, *operands, optimize=True)
```

**优化模式分析**：
- **`tnm,tm->tn`**：SDE 中的常见模式（时间 × 模式 × 噪声）
  - 三维张量与二维张量 contraction
  - NumPy 优化后仍较慢，JIT 可加速 5-10 倍
- **`tm,mk->tk`**：矩阵乘法（时间 × 模式 × 模式）
  - 大批量矩阵乘法，Numba JIT 减少 Python 开销

**JIT 配置**：
```python
@njit(cache=True, fastmath=False)
```
- `cache=True`：编译结果缓存到磁盘，避免重复编译
- `fastmath=False`：严格 IEEE 754 浮点运算，保证数值精度（科学计算必需）

**Numba RNG 适配**（`_NumbaRNG`）：
```python
class _NumbaRNG:
    """轻量级适配器，满足 RNGBase 语义"""

    def __init__(self, seed: int | np.random.SeedSequence | None):
        if isinstance(seed, np.random.SeedSequence):
            self._seed_seq = seed
        elif seed is None:
            self._seed_seq = np.random.SeedSequence()  # 从熵获取种子
        else:
            self._seed_seq = np.random.SeedSequence(seed)
        self._gen = np.random.default_rng(self._seed_seq)
```

**设计权衡**：
- **预编译 vs 动态**：特定模式预编译（高性能），一般情况回退（灵活性）
- **单线程 vs 并行**：当前单线程编译（简化实现），未来可启用 `prange`
- **缓存 vs 编译时间**：启用缓存增加首次启动时间，但后续运行更快

### 6.5 CuPy Backend - 实验性 GPU 加速

**定位**：实验性 CUDA 后端，为大规模 GPU 计算提供路径。

**模块加载**：
```python
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False
    # 创建空壳类，访问时抛出明确错误
    class _CPShim:
        def __getattr__(self, name):
            raise ImportError("cupy is required for the CuPy backend")
    cp = _CPShim()
```

**容错设计**：
- **延迟错误**：导入时不报错，使用时（访问属性）才报错
- **空壳类**：避免整个模块无法加载，不影响其他后端

**CuPy RNG**（`_CuPyRNG`）：
```python
class _CuPyRNG:
    """CuPy RandomState 的适配器"""

    def __init__(self, seed: int | None):
        # 使用 NumPy SeedSequence 确保种子可移植性
        if seed is None:
            self._rs = cp.random.RandomState()
        else:
            self._rs = cp.random.RandomState(int(seed))

    def seed(self, value: int | None) -> None:
        if value is None:
            self._rs = cp.random.RandomState()
        else:
            self._rs = cp.random.RandomState(int(value))

    def spawn(self, n: int) -> list["_CuPyRNG"]:
        # 使用 NumPy SeedSequence 派生稳定整数种子
        ss = np.random.SeedSequence()
        children = ss.spawn(n)
        return [_CuPyRNG(int(child.generate_state(1, dtype=np.uint64)[0]))
                for child in children]
```

**设备检测**：
```python
def device(self) -> str | None:
    try:
        dev = cp.cuda.runtime.getDevice()
        return f"cuda:{dev}"  # 多 GPU 支持
    except Exception:
        return "cuda"  # 默认 CUDA 设备
```

**数组创建**：
```python
def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
    return cp.zeros(shape, dtype=dtype)  # 直接在 GPU 上创建

def copy(self, x: Any) -> Any:
    return cp.array(x, copy=True)  # GPU 上的深拷贝
```

**能力报告**：
```python
def capabilities(self) -> dict:
    return {
        "device": self.device(),
        "optimized_contractions": True,
        "supports_complex_view": False,  # CuPy 限制（类似 NumPy）
        "real_imag_split": True,
        "stack": True,
        "to_device": True,
        "cupy": _CUPY_AVAILABLE,  # 报告 CuPy 是否可用
    }
```

**实验性质**：
- **API 简化**：CuPy 与 NumPy API 高度一致，实现相对简单
- **内存管理**：CuPy 自动处理 GPU 内存，避免手动 `cudaMemcpy`
- **限制**：某些高级功能（如复数视图）尚未支持

### 6.6 能力检测机制 - capabilities() 方法

**设计动机**：
后端能力差异较大（GPU/CPU、JIT/解释、特殊优化），需要运行时查询机制。

**能力键值对**：
```python
{
    "device": "cuda:0",              # 当前设备或 None（CPU）
    "optimized_contractions": True,   # 支持 contraction 优化（JIT/加速库）
    "supports_complex_view": False,   # 支持复数视图（零拷贝 real/imag）
    "real_imag_split": True,          # 支持实部/虚部分离
    "stack": True,                    # 支持 stack() 方法
    "to_device": False,               # 支持设备迁移
    "numpy": False,                   # 底层是否使用 NumPy
}
```

**使用示例**：
```python
# 检查是否支持设备迁移
if backend.capabilities().get("to_device"):
    x_gpu = backend.to_device(x, "cuda:0")
else:
    x_gpu = x  # CPU 回退

# 检查是否支持复数视图
if backend.capabilities().get("supports_complex_view"):
    real_part = backend.real(x)
else:
    real_part = backend.mean([x, -1j * x])  # 替代实现
```

**设计优势**：
- **运行时决策**：根据实际能力选择最优代码路径
- **优雅降级**：不支持某功能时选择备选实现
- **避免异常**：不依赖 try/except 检查特性（性能友好）
- **调试友好**：查询后端能力，快速定位兼容性问题

### 6.7 后端选择策略

**配置驱动选择**：
```yaml
# global.yaml
plugins:
  backend:
    name: "torch"  # 或 "numpy"、"numba"、"cupy"
    params:
      device: "cuda"  # 对于 torch/cupy
```

**动态选择**：
```python
# 根据可用性和需求选择后端
if torch.cuda.is_available():
    backend = TorchBackend()
elif numba_available:
    backend = NumbaBackend()
else:
    backend = NumpyBackend()
```

**性能权衡**：
- **NumPy**：通用性强、稳定、调试方便，性能中等
- **Numba**：CPU JIT 加速，适合大规模数值计算，专用优化路径
- **PyTorch**：GPU 加速、自动微分、动态图，适合研究
- **CuPy**：纯 GPU 计算，大规模并行，适合生产环境

### 6.8 Backend 系统设计原则

**1. 接口统一性**：
- 所有后端实现相同的方法签名
- 行为语义保持一致（除非底层库限制）
- 返回值类型和形状严格匹配

**2. 性能优先**：
- `asarray()` 零拷贝（尽可能）
- `copy=False` 避免不必要的数据转换
- JIT 和 GPU 加速专用路径

**3. 错误容错**：
- 可选方法用 `hasattr()` 检查
- 设备迁移失败时返回原对象
- 延迟错误而非早期失败

**4. 依赖隔离**：
- 算法代码只依赖 BackendBase，不依赖具体库
- 可选依赖（CuPy、PyTorch）用 try/except 处理
- 回退机制保证无依赖也能运行

**5. 设备感知**：
- 自动检测 CPU/GPU 设备
- 设备特定的数组创建和 RNG
- 支持设备迁移（PyTorch/CuPy）

**6. 扩展友好**：
- Protocol 定义清晰的扩展点
- 新后端只需实现必需方法
- 能力检测支持渐进增强
