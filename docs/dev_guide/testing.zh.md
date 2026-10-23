---
description: 测试分层与 CI 闸门
---

# 测试分层与 CI 闸门

测试套件按五个 pytest marker 分层，声明在 `pyproject.toml` 中并由
`--strict-markers` 强制。目标是每个用例都明确自己所属的层，使"快闸门"与"全量"
从同一棵树上选取。

## Marker

| Marker | 含义 |
| :-- | :-- |
| `unit` | 快速、隔离：单个组件，不跑 scheduler。 |
| `integration` | 跨组件 / 跨插件行为：scheduler、engine、analyser 图。 |
| `e2e` | 完整 CLI / workflow 运行，含持久化的 Session 与 Job 目录。 |
| `gpu` | 需要 CUDA 设备（CuPy）；无设备时自动跳过。 |
| `slow` | 长耗时用例；快闸门中应排除。 |

两条规则保证分层一致：

- 未显式标注层 marker 的用例会被自动标为 `unit`
  （`tests/conftest.py::pytest_collection_modifyitems`）；因此把文件标为
  `integration`/`e2e` 就等于把它移出快闸门。
- `gpu` 与 `slow` 与层正交：integration 文件里的 GPU 用例按用例标
  `@pytest.mark.gpu`（不是按文件），spawn 子进程的用例再叠加 `slow`。

## CUDA 门控

`tests/conftest.py::cuda_available()` 是全套件唯一的设备探测：它返回 CuPy 能否
访问至少一个 CUDA 设备。它有两处用途：

- `gpu` marker 门控：无设备时跳过所有含 `gpu` 关键字的用例；
- 用例级 `@pytest.mark.skipif(not cuda_available(), ...)`，用于即便漏标 marker
  也需要设备的用例。

不要在测试模块里再写私有设备探测：从 `tests.conftest` 导入 `cuda_available`。

## 本地常用命令

```bash
pytest -m "unit and not gpu"                          # 快闸门
pytest -m "integration and not gpu and not slow"      # integration 层
pytest -m "e2e and not gpu"                           # CLI / workflow 运行
pytest -m "slow and not gpu"                          # 长耗时回归扫描
pytest -m "not gpu"                                   # 所有可在 CPU 上跑的用例
pytest                                                # 全量
```

## CI 各 job

`.github/workflows/ci.yml` 每层一个 job：

| Job | 选择表达式 | 解释器 | timeout |
| :-- | :-- | :-- | :-- |
| `test` | `-m "unit and not gpu"` | 3.11 / 3.12 / 3.13 / 3.14 | 20 分钟 |
| `integration` | `-m "integration and not gpu and not slow"` | 3.13 | 45 分钟 |
| `e2e` | `-m "e2e and not gpu"`（仅 push，阻塞） | 3.13 | 30 分钟 |
| `slow` | `-m "slow and not gpu"`（仅 push） | 3.13 | 60 分钟 |
| `lint` | `ruff check .` 与 `mypy packages` | 3.13 | 15 分钟 |

所有 job 都安装纯 CPU 依赖组（`uv sync --no-dev --group dev-cpu`）：它带 pytest、
ruff、mypy 与 CLI/GUI/资源包，但不含 CuPy。这样整个解释器矩阵在没有 CuPy wheel 的
版本上也能安装，并且套件始终走"未安装 CuPy"的分支——所有 `gpu` 用例经由设备探测被跳过，
而不是经由 import 失败。

`mypy` 目前只检查 `packages`。tests 由 `ruff` 检查静态问题，但尚未纳入类型检查：
`mypy packages tests` 仍报 74 个已登记的存量错误，因此测试侧类型闸门暂挂。

## GPU CI lane

`.github/workflows/gpu.yml` 是唯一真正执行 `gpu` 用例的 workflow，它跑在自建的 CUDA
runner 上而不是 GitHub 托管 runner 上。`-m gpu` 当前在 13 个文件里收集到 49 个参数化用例。

- Runner：注册一台带 `self-hosted` 与 `gpu` 标签的 self-hosted runner，机器装好 NVIDIA
  驱动即可。CuPy 14 起所需的 CUDA 组件以 PyPI wheel 形式提供
  （`cupy-cuda12x[ctk]`），不需要系统 CUDA toolkit。
- 触发：目前只有 `workflow_dispatch`。没有匹配的 runner 时，push 触发的 job 会先排队约
  24 小时再失败，因此只有首次手动 dispatch 通过后才把 `push` 触发器加回来。
- 安装：直接 `uv sync`（默认 `dev` 组，含 `qphase[cupy]`）。该 lane 绝不能改用
  `dev-cpu`：没有 CuPy 时所有用例都会 skip，lane 会看起来是绿的却什么都没测。
- 保险：单独一步在 CuPy 无法 import 或
  `cupy.cuda.runtime.getDeviceCount()` 为 0 时直接让 job 失败，避免"全部 skip 即通过"。
- 矩阵：3.11–3.13。等 `uv.lock` 从 `cupy-cuda12x` 13.6.0（只有 cp311–cp313 wheel）
  升级到 CuPy 14（有 cp314 wheel）后再加入 3.14。
