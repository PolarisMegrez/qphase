---
description: Backend System
---

# Backend System

The **Backend System** serves as the computational abstraction layer for QPhase. It addresses the challenge of hardware heterogeneity by providing a unified interface for tensor operations, enabling simulation code to remain agnostic to the underlying computational library (e.g., NumPy, PyTorch, CuPy).

## Architectural Objective

The primary objective is to decouple the **Physical Model** from the **Computational Implementation**. This allows a single model implementation to:
1.  Execute on CPUs for debugging or small-scale simulations (via NumPy).
2.  Execute on GPUs for high-performance parallel simulations (via PyTorch or CuPy).
3.  Support automatic differentiation (via PyTorch/JAX) without code modification.

## The Backend Protocol

The core of this system is the `BackendBase` Protocol, which defines the contract for all computational backends. It standardizes the API for array creation, linear algebra, random number generation, and common utility functions.

**Strict Hardware Agnosticism**: The protocol is designed to prevent "host transfers" (moving data between CPU and GPU) during the simulation loop. All operations, including reshaping and indexing, must be performed via the backend interface.

```python
@runtime_checkable
class BackendBase(Protocol):
    """Abstract interface for tensor operations."""

    # Identification
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...

    # Array Creation
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def arange(self, start: int, stop: int | None = None, step: int = 1, dtype: Any | None = None) -> Any: ...

    # Shape Manipulation
    def expand_dims(self, x: Any, axis: int) -> Any: ...
    def repeat(self, x: Any, repeats: int, axis: int | None = None) -> Any: ...
    def stack(self, arrays: tuple[Any, ...], axis: int = 0) -> Any: ...
    def concatenate(self, arrays: tuple[Any, ...], axis: int = -1) -> Any: ...

    # Math & Linear Algebra
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...
    def isnan(self, x: Any) -> Any: ...

    @property
    def pi(self) -> float: ...

    # Random Number Generation
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
```

## Implementation Strategy

### Wrapper Pattern
Concrete backends (e.g., `NumpyBackend`, `TorchBackend`) implement the `BackendBase` protocol by wrapping the respective library calls. This ensures that method signatures (arguments, return types) are consistent across all implementations, smoothing over API differences between libraries (e.g., `np.concatenate` vs `torch.cat`, `np.repeat` vs `torch.repeat_interleave`).

### Tensor Dispatching
In the simulation kernel, the backend instance is typically injected as `self.xp` (following the array API standard convention). All mathematical operations are dispatched through this instance.

**Example:**
```python
# Hardware-agnostic implementation
def drift(self, state):
    # self.xp could be numpy or torch
    # Using self.xp.pi ensures we use the correct scalar type if needed
    phase = 2.0 * self.xp.pi * state
    return -1j * self.xp.einsum("ij,j->i", self.hamiltonian, phase)
```

## Available Backends

### 1. NumPy Backend (`backend: numpy`)
*   **Target**: CPU.
*   **Use Case**: Development, debugging, small-scale simulations.
*   **Characteristics**: Double precision by default, deterministic execution, broad compatibility.

### 2. PyTorch Backend (`backend: torch`)
*   **Target**: CPU / GPU (CUDA/MPS).
*   **Use Case**: Large-scale parallel simulations, gradient-based optimization.
*   **Characteristics**: Supports `float32`/`float64`, automatic device management, batched operations. Implements `repeat` using `torch.repeat_interleave`.

### 3. CuPy Backend (`backend: cupy`)
*   **Target**: NVIDIA GPU.
*   **Use Case**: High-performance computing where PyTorch overhead is undesirable.
*   **Characteristics**: NumPy-compatible API on GPU.

## Extending the Backend

Developers can introduce support for new libraries (e.g., JAX, TensorFlow) by implementing a class that satisfies the `BackendBase` protocol and registering it under the `backend` namespace.
