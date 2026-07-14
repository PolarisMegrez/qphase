# 短期改进计划

> **已归档**：本计划中的具体工作项已基本完成。当前活跃计划见 `C:\Users\HP\.kimi\plans\ms-marvel-multiple-man-beta-ray-bill.md`。


本计划只记录短期内可实现、适合逐步提交的业务逻辑改进。大幅度架构重构建议不放入本文件。

## 1. 固化配置加载与参数扫描契约

- 为 `configs/jobs/*.yaml` 的顶层插件写法建立回归测试，覆盖 `backend`、`integrator`、`model`、`analyser` 被正确抽取到 `JobConfig.plugins`。
- 为 `model` 参数扫描补充 smoke test，确认 `kerr_3pa.epsilon`、`kerr_3mode.kappa_a`、`vdp_2mode.omega_a` 这类列表参数会按系统配置展开。
- 检查 `JobExpander`、`Scheduler._validate_job_dependencies`、`Scheduler._run_job` 是否始终使用同一份插件解析结果，避免“加载时识别、验证时漏看、运行时又重建”的分叉。
- 在文档中明确 job 文件推荐格式：顶层插件段是推荐写法，`plugins:` 是兼容写法。

## 2. 提升后处理入口的健壮性

- 为 `qphase postprocess` 增加 `--dry-run` 或 `--inspect` 模式，只列出将处理的 `.npz`、扫描参数值、PSD key、mode 和输出路径，不写文件。
- 为 Lorentz 拟合结果增加可配置质量判据，例如最小 `R2`、最小峰高、最大线宽；不满足时保留行但标记 `status=low_quality`。
- 支持用户选择 PSD 频率轴范围：全轴、正频率、负频率或显式 `--freq-min/--freq-max`，避免批处理中临时裁剪数组。
- 为 `dist_merged.npz`、`pdist_merged.pkl` 写入 schema/version 字段，先把 experimental 输出的最小兼容边界固定下来。

## 3. 准备 CPU 友好的示例与 smoke 数据

- 新增小规模 CPU job fixture，使用 `backend.numpy`、短时间、少轨迹，专门用于 CLI smoke 和文档示例。
- 保留 GPU/CuPy 科研配置，但避免 README 和基础文档默认要求 CUDA。
- 为 `qphase run jobs <cpu-demo>` 到 `qphase postprocess <run-dir>` 建立一个端到端 smoke 流程，输出不追求物理精度，只验证契约。

## 4. 收敛 PSD 寻峰与 Lorentz 拟合边界

- 明确 `PsdAnalyzer.find_peaks` 是单 job 内部寻峰，`qphase postprocess` 是跨 job 谱线拟合与导出，两者字段命名避免混淆。
- 为 `peak_finding` 的 `scipy` 和 `rational` 方法补最小数值测试，覆盖无峰、单峰、多峰、失败回退路径。
- 检查 `PeakInfo.properties` 中的 ndarray/complex 是否需要在保存前转换为可 pickle/JSON 友好的结构。

## 5. 清理过时模型与测试引用

- 删除或更新所有引用已移除模型 `kerr_cavity`、`vdp_level2`、`vdp_two_mode` 的测试、文档和配置。
- 保证 `models/.qphase_plugins.yaml`、`models/__init__.py`、`configs/jobs/*.yaml` 中的模型 key 与模型类 `name` 一致。
- 为当前保留模型 `kerr_3pa`、`kerr_3mode`、`vdp_2mode` 增加轻量导入/实例化测试。

## 6. 文档提交前检查清单

- 每次修改 CLI 或输出格式后运行 `mkdocs build --strict`，确保中英文导航和链接同步。
- README 只保留当前可运行的命令形态：`qphase run jobs ...` 与 `qphase postprocess ...`。
- `docs/user_guide/output*.md` 与 `docs/api/sde*.md` 应同步说明 `.npz` 字段、`analysis` schema 和后处理输出 bundle。
