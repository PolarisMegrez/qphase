---
layout: default
title: 6 Backend 系统 - 统一计算接口与多后端兼容
---

# 6 Backend 系统 - 统一计算接口与多后端兼容

### 6.0 设计目标与架构

Backend 系统是 qphase 核心的"计算抽象层"，通过 BackendBase Protocol 定义统一的计算接口，解决多后端兼容问题。

**核心动机**：
- **解耦计算与算法**：算法代码只依赖 Backend 接口，不依赖具体库
- **无缝切换后端**：通过配置选择 NumPy/Numba/PyTorch/CuPy，无需修改算法
- **性能优化**：不同场景可选择不同后端（CPU/GPU、JIT/解释执行）
- **依赖隔离**：算法代码不直接引入 NumPy、PyTorch 等重型依赖

**BackendBase 协议定义**：
```python
@runtime_checkable
class BackendBase(Protocol):
    """Minimal backend protocol for array ops, linalg, RNG, and helpers."""

    # 识别信息
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...
    def capabilities(self) -> dict[str, Any]: ...

    # 数组创建与转换
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def empty_like(self, x: Any) -> Any: ...
    def copy(self, x: Any) -> Any: ...

    # 数学运算
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...

    # 随机数生成
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def spawn_rngs(self, master_seed: int, n: int) -> list[Any]: ...

    # 复杂数支持
    def real(self, x: Any) -> Any: ...
    def imag(self, x: Any) -> Any: ...
    def abs(self, x: Any) -> Any: ...
    def mean(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any: ...

    # FFT
    def fft(self, x: Any, axis: int = -1, norm: Literal["backward", "ortho", "forward"] | None = None) -> Any: ...
    def fftfreq(self, n: int, d: float = 1.0) -> Any: ...
```

**设计原则**：
- **最小接口**：只定义必需的 20+ 个方法，覆盖 80% 常用计算场景
- **领域无关**：不包含 SDE 特定概念，通用性适用于所有科学计算
- **可选方法**：`stack()` 和 `to_device()` 用 `hasattr()` 检查是否支持
- **类型一致**：所有方法接受和返回相同抽象类型（Any），避免泛型约束

### 6.1 核心接口分类 - 统一计算抽象

**1. 识别与元数据**：
```python
def backend_name(self) -> str: ...
def device(self) -> str | None: ...
def capabilities(self) -> dict[str, Any]: ...
```
- `backend_name()`：返回后端标识符（"numpy"、"torch" 等）
- `device()`：返回设备字符串（"cpu"、"cuda:0"、None）
- `capabilities()`：报告支持的特性和环境标志

**2. 数组创建与转换**：
```python
def asarray(x, dtype=None)        # 统一转换（推荐）
def array(x, dtype=None)          # 创建副本
def zeros(shape, dtype)           # 零数组
def empty(shape, dtype)           # 未初始化数组
def empty_like(x)                 # 仿制数组
def copy(x)                       # 深拷贝
```
- `asarray()`：不复制直接转换（如果可能），性能最佳
- `array()`：始终创建新数组
- `empty_like()`：使用输入数组的形状和 dtype，但内容未初始化

**3. 数学运算**：
```python
def einsum(subscripts, *operands)      # 爱因斯坦求和（高效张量运算）
def concatenate(arrays, axis=-1)       # 数组连接
def cholesky(a)                        # Cholesky 分解
```
- `einsum()`：使用爱因斯坦求和记号表示张量运算，高度紧凑
  - 示例：`einsum("ijk,kl->ijl", A, B)` 表示矩阵乘法
  - 比 `np.matmul()` 更灵活，支持高维张量 contraction

**4. 随机数生成**：
```python
def rng(seed)                # 创建 RNG 句柄
def randn(rng, shape, dtype) # 标准正态分布采样
def spawn_rngs(seed, n)      # 生成 n 个独立 RNG
```
- **句柄模式**：rng() 返回一个可传递的 RNG 对象（非纯函数）
- **确定性**：相同 seed 产生相同序列，保证可重现性
- **独立流**：spawn_rngs() 生成多个独立 RNG（多轨迹模拟）

**5. 复杂数支持**：
```python
def real(x) / imag(x)  # 复数实部/虚部视图
def abs(x)             # 模长
def mean(x, axis)      # 平均值
```
- `real()`/`imag()`：返回视图而非副本（零拷贝）
- 支持沿指定轴计算均值

**6. FFT 操作**：
```python
def fft(x, axis=-1, norm=None)  # 快速傅里叶变换
def fftfreq(n, d=1.0)           # FFT 频率网格
```
- 与 NumPy 的 FFT API 保持一致
- norm 参数控制归一化方式（"backward"、"ortho"、"forward"）

### 6.2 NumPy Backend - 参考 CPU 实现

**定位**：作为兼容性基线和性能参考，所有其他后端应与之行为一致。

**核心实现**：
```python
class NumpyBackend(Backend):
    def backend_name(self) -> str:
        return "numpy"

    def device(self) -> str | None:
        return None  # CPU，无设备概念

    def asarray(self, x: Any, dtype: Any | None = None) -> Any:
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        return np.zeros(shape, dtype=dtype)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        # 启用 NumPy 的 contraction 路径优化
        return np.einsum(subscripts, *operands, optimize=True)
```

**RNG 实现**：
```python
def rng(self, seed: int | None) -> Any:
    return np.random.default_rng(seed)

def spawn_rngs(self, master_seed: int, n: int) -> list[Any]:
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n)
    return [np.random.default_rng(child) for child in children]

def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    out = rng.normal(size=shape)
    # 类型转换：避免复制（copy=False）
    return out.astype(dtype if dtype is not None else np.float64, copy=False)
```

**设计细节**：
- **SeedSequence**：NumPy 的现代种子管理，支持派生独立流
- **copy=False**：类型转换时尽可能视图共享，避免内存拷贝
- **einsum 优化**：`optimize=True` 启用子表达式消除，性能提升 10-100%

**能力报告**：
```python
def capabilities(self) -> dict:
    return {
        "device": None,
        "optimized_contractions": False,  # 无 JIT 优化
        "supports_complex_view": False,   # 复数视图（NumPy 的限制）
        "real_imag_split": True,          # 实部/虚部分离
        "stack": True,
        "to_device": False,               # CPU 后端不支持设备迁移
        "numpy": True,
    }
```

### 6.3 PyTorch Backend - 设备感知与复杂类型

**定位**：支持 CPU/GPU 的通用后端，动态设备选择，复杂类型映射。

**设备检测**：
```python
def device(self) -> str | None:
    try:
        import torch as torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return f"cuda:{idx}"  # 多 GPU 支持
    except Exception:
        return None
    return "cpu"
```

**类型映射**（`_to_torch_dtype()`）：
```python
def _to_torch_dtype(dtype: Any | None):
    """将 Python/NumPy dtypes 映射到 torch dtypes"""
    if dtype is None:
        return None
    try:
        import numpy as _nplocal
        import torch as torch

        if dtype is complex or str(dtype) == "complex":
            return torch.complex128
        if dtype is float or str(dtype) in ("float", "float64"):
            return torch.float64
        # ... 更多映射
    except Exception:
        return dtype
    return dtype
```

**类型映射逻辑**：
- **Python 原生类型**：`complex` → `torch.complex128`、`float` → `torch.float64`
- **NumPy dtypes**：`np.complex128` → `torch.complex128`、`np.float64` → `torch.float64`
- **容错设计**：映射失败时返回原始 dtype（假设已兼容）

**GPU 数组创建**：
```python
def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
    import torch as torch
    dev = self.device() or "cpu"  # 默认 CPU
    td = _to_torch_dtype(dtype)
    return torch.zeros(*shape, dtype=td, device=dev)

def asarray(self, obj: Any, dtype: Any | None = None) -> Any:
    import torch as torch
    td = _to_torch_dtype(dtype)
    t = torch.as_tensor(obj, dtype=td)
    return t
```

**Torch RNG 适配**（`_TorchRNG`）：
```python
class _TorchRNG:
    """Torch Generator 的适配器，提供标准 RNG 接口"""

    def __init__(self, seed: int | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._gen = torch.Generator(device=self.device)  # 设备特定生成器
        if seed is None:
            seed = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0])
        self._gen.manual_seed(int(seed))

    def seed(self, value: int | None) -> None:
        if value is None:
            value = int(_np.random.SeedSequence().generate_state(1, dtype=_np.uint64)[0])
        self._gen.manual_seed(int(value))

    def spawn(self, n: int) -> list["_TorchRNG"]:
        ss = _np.random.SeedSequence()
        children = ss.spawn(n)
        return [_TorchRNG(int(c.generate_state(1, dtype=_np.uint64)[0]), device=self.device)
                for c in children]

    @property
    def generator(self):
        return self._gen  # 暴露底层生成器
```

**设计优势**：
- **设备特定 RNG**：每个设备有独立的 Generator，避免 GPU/CPU 同步开销
- **种子管理**：使用 NumPy SeedSequence 确保种子序列兼容性
- **spawn() 优化**：为每个设备创建独立的 RNG，保持设备一致性

**randn 实现**：
```python
def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    import torch as torch
    g = cast(_TorchRNG, rng).generator  # 获取底层 Generator
    dev = self.device() or "cpu"
    t = torch.randn(*shape, generator=g, device=dev)  # 在正确设备上生成
    td = _to_torch_dtype(dtype)
    return t.to(dtype=td) if td is not None else t
```

**设备迁移**（`to_device()`）：
```python
def to_device(self, x: Any, device: str | None) -> Any:
    if device is None:
        return x
    try:
        return x.to(device)  # 迁移到目标设备
    except Exception:
        return x  # 迁移失败时返回原对象（容错）
```

**能力报告**：
```python
def capabilities(self) -> dict:
    return {
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
