---
layout: default
title: Installation
parent: User Guide
nav_order: 1
---

# Installation

This guide covers the installation process for QPhase and its official plugins.

## Prerequisites

*   **Operating System**: Linux, macOS, or Windows (WSL2 recommended for performance).
*   **Python**: Version 3.10 or higher.
*   **Git**: For cloning the repository.

## Recommended: Virtual Environment

We strongly recommend using a virtual environment to avoid conflicts with other Python packages.

### Using Conda (Recommended)

```bash
conda create -n qphase python=3.11
conda activate qphase
```

### Using venv

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

## Installation from Source

Currently, QPhase is installed directly from the source code.

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/PolarisMegrez/qphase.git
    cd qphase
    ```

2.  **Install Core Package**

    Install the core framework in editable mode (`-e`), which allows you to modify the code without reinstalling.

    ```bash
    pip install -e packages/qphase
    ```

    For most users, we recommend installing with the `standard` optional dependencies which include commonly used packages:

    ```bash
    pip install -e packages/qphase[standard]
    ```

    This includes Numba (for JIT compilation) and PyTorch (for GPU acceleration).

3.  **Install Official Plugins**

    QPhase is modular. You can install only the components you need.

    *   **SDE Engine** (Stochastic Differential Equations):
        ```bash
        pip install -e packages/qphase_sde
        ```

    *   **Visualization Tools**:
        ```bash
        pip install -e packages/qphase_viz
        ```

4.  **Install Dependencies**

    If you have a `requirements.txt` file in the root, you can install it, but usually, the package installation above handles dependencies automatically.

    ```bash
    # Optional: Install development dependencies (testing, docs)
    pip install -r requirements.txt
    ```

## Verification

To verify that the installation was successful, run the following command:

```bash
qps --version
```

You should see the version number of the installed QPhase core.

To check which plugins are detected:

```bash
qps list
```

This will display a list of registered backends, engines, and models.

## Troubleshooting

### "Command not found: qps"

Ensure your Python environment's `bin` (or `Scripts` on Windows) directory is in your system's PATH. If you installed in a virtual environment, make sure it is activated.

### "ModuleNotFoundError: No module named 'qphase'"

This usually happens if you installed dependencies but forgot to install the package itself. Run `pip install -e packages/qphase` again.
