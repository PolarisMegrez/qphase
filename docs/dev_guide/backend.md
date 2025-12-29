---
layout: default
title: Backend System
parent: Developer Guide
nav_order: 3
---

# Backend System

The **Backend System** serves as the computational abstraction layer for QPhase. It addresses the challenge of hardware heterogeneity by providing a unified interface for tensor operations, enabling simulation code to remain agnostic to the underlying computational library (e.g., NumPy, PyTorch, CuPy).

## Architectural Objective

The primary objective is to decouple the **Physical Model** from the **Computational Implementation**. This allows a single model implementation to:
1.  Execute on CPUs for debugging or small-scale simulations (via NumPy).
2.  Execute on GPUs for high-performance parallel simulations (via PyTorch or CuPy).
3.  Support automatic differentiation (via PyTorch/JAX) without code modification.

## The Backend Protocol

The core of this system is the `BackendBase` Protocol, which defines the contract for all computational backends. It standardizes the API for array creation, linear algebra, and random number generation.

```python
@runtime_checkable
class BackendBase(Protocol):
    """Abstract interface for tensor operations."""

    # Identification
    def backend_name(self) -> str: ...
    def device(self) -> str | None: ...

    # Array Operations
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...

    # Linear Algebra
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...

    # Random Number Generation
    def rng(self, seed: int | None) -> Any: ...
    def randn(self, rng: Any, shape: tuple[int, ...], dtype: Any) -> Any: ...
```

## Implementation Strategy

### Wrapper Pattern
Concrete backends (e.g., `NumpyBackend`, `TorchBackend`) implement the `BackendBase` protocol by wrapping the respective library calls. This ensures that method signatures (arguments, return types) are consistent across all implementations, smoothing over API differences between libraries (e.g., `np.concatenate` vs `torch.cat`).

### Tensor Dispatching
In the simulation kernel, the backend instance is typically injected as `self.xp` (following the array API standard convention). All mathematical operations are dispatched through this instance.

**Example:**
```python
# Hardware-agnostic implementation
def drift(self, state):
    # self.xp could be numpy or torch
    return -1j * self.xp.einsum("ij,j->i", self.hamiltonian, state)
```

## Available Backends

### 1. NumPy Backend (`backend: numpy`)
*   **Target**: CPU.
*   **Use Case**: Development, debugging, small-scale simulations.
*   **Characteristics**: Double precision by default, deterministic execution, broad compatibility.

### 2. PyTorch Backend (`backend: torch`)
*   **Target**: CPU / GPU (CUDA/MPS).
*   **Use Case**: Large-scale parallel simulations, gradient-based optimization.
*   **Characteristics**: Supports `float32`/`float64`, automatic device management, batched operations.

### 3. CuPy Backend (`backend: cupy`)
*   **Target**: NVIDIA GPU.
*   **Use Case**: High-performance computing where PyTorch overhead is undesirable.
*   **Characteristics**: NumPy-compatible API on GPU.

## Extending the Backend

Developers can introduce support for new libraries (e.g., JAX, TensorFlow) by implementing a class that satisfies the `BackendBase` protocol and registering it under the `backend` namespace.
