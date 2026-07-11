# QPhase 下一步更新方案

> **已归档**：本计划中的大部分工作项已在 2026-07 前完成，后续架构整顿见 `C:\Users\HP\.kimi\plans\ms-marvel-multiple-man-beta-ray-bill.md`。


基于对 docs、README、根目录 `plan.md`、`packages`、`configs`、`tests` 的梳理，汇总当前进度并给出下一步更新计划。

---

## 一、当前进度总览（对照 `plan.md`）

| 计划项 | 完成度 | 关键结论 |
|--------|--------|----------|
| **1. 固化配置加载与参数扫描契约** | ~60% | 加载/运行阶段已能识别顶层插件段（`backend/integrator/model/analyser`），但 `Scheduler._validate_job_dependencies()` 只看 `job.plugins`，未回退全局默认值，与 `_run_job()` 口径不一致；缺少顶层插件段回归测试与真实模型参数扫描 smoke。 |
| **2. 提升后处理入口的健壮性** | ~30% | `qphase postprocess` 基础可用，但 `--dry-run/--inspect`、频率范围 `--freq-min/--freq-max`、Lorentz 拟合质量判据、`dist_merged.npz`/`pdist_merged.pkl` 的 schema/version 均未实现。 |
| **3. 准备 CPU 友好的示例与 smoke 数据** | ~20% | 现有 `configs/jobs/*.yaml` 全用 `backend.cupy`；无 `backend.numpy` 的小规模示例；`tests/data/*.yaml` 仍引用已删除模型且字段过时。 |
| **4. 收敛 PSD 寻峰与 Lorentz 拟合边界** | ~50% | 单 job 寻峰（`PeakInfo`）与跨 job 拟合（`LorentzFitResult`）字段已区分；但寻峰方法缺少无峰/单峰/多峰/失败测试，`PeakInfo.properties` 中的 ndarray/complex 未统一转换。 |
| **5. 清理过时模型与测试引用** | ~50% | 源码与注册表已清理；但 `tests/data/*.yaml`、多份文档、CLI 帮助文本仍引用 `vdp_two_mode`/`kerr_cavity`；缺少保留模型的轻量实例化测试。 |
| **6. 文档提交前检查清单** | ~40% | `mkdocs build --strict` 可通过；但 README 与多数文档 CLI 形态不一致（`qphase run jobs ...` vs `qphase run ...`），文档示例含旧模型名和错误字段 `t_end`。 |

---

## 二、推荐的两阶段更新路线

### 第一阶段：清理不一致 + 补齐 smoke

**目标**：让所有文档、CLI 帮助、测试数据与当前代码一致，并建立可快速运行的 CPU smoke，为后续改动提供快速回归。

1. **统一并修正 CLI 命令形态**
   - 在 `packages/qphase/qphase/commands/run.py` 的帮助文本与示例中：
     - 把所有 `qphase run jobs ...` 改为 `qphase run ...`；
     - 把过时的 `t_end`/`n_steps` 改为当前字段 `t1`/`dt`/`n_traj`。
   - 同步更新 `README.md`、`docs/user_guide/getting_started.md(.zh)`、`docs/user_guide/configuration.md(.zh)`、`docs/api/cli.md(.zh)` 中的命令示例。
   - 明确取舍：当前代码实现是 `qphase run <job>`，不把 `jobs` 作为子命令；文档与 README 统一到此形态。

2. **清理过时模型引用**
   - 删除或重写 `tests/data/test_job_sde.yaml`、`test_param_scan_cartesian.yaml`、`test_param_scan_zipped.yaml` 中的 `vdp_two_mode`，改用当前模型 `vdp_level3` 或 `kerr_3pa`，并修正字段（`t_end`→`t1`、`dt`、`n_traj`；`euler`→`euler_maruyama`）。
   - 更新 `docs/user_guide/configuration.md(.zh)`、`docs/api/cli.md(.zh)`、`docs/dev_guide/registry.md(.zh)` 中的 `vdp_two_mode`/`kerr_cavity` 示例，改为 `kerr_3pa` / `vdp_level3`。
   - 清理 `models/__pycache__` 中旧模型编译缓存（可手动删除，或加入清理规则）。

3. **新增 CPU smoke 示例与端到端测试**
   - 新增 `configs/jobs/cpu_smoke_kerr_3pa.yaml`：
     - `backend.numpy`，`t1: 1.0`，`dt: 0.01`，`n_traj: 2`，`save: true`；
     - `analyser.psd.modes: [0]`；
     - `model.kerr_3pa.epsilon: [0.025, 0.05]`（极短扫描）。
   - 新增测试 `tests/qphase/test_cpu_smoke.py`：
     - 使用上述 job 运行 `qphase run cpu_smoke_kerr_3pa`；
     - 断言运行成功、生成 `runs/<timestamp>_*` 目录、目录下存在 `.npz` 文件；
     - 对同一目录执行 `qphase postprocess ... --scan-param epsilon --mode 0`，断言生成 `fit_results.csv` 与 `psd_merged.csv`。
   - 新增 `tests/qphase/test_plugin_section_extraction.py`：
     - 验证顶层 `backend/integrator/model/analyser` 与兼容 `plugins:` 写法都被加载到 `JobConfig.plugins`；
     - 验证 `Scheduler._validate_job_dependencies()` 与 `_run_job()` 使用同一份插件解析结果。

4. **修复 Scheduler 验证与运行的插件口径不一致**
   - 修改 `packages/qphase/qphase/core/scheduler.py::_validate_job_dependencies()`：
     - 合并 `job.plugins` 与 `job.model_extra` 中的插件命名空间；
     - 若引擎 required plugin 可由全局默认配置提供，则不应抛 `missing required plugins`。
   - 保持“可选插件不继承全局默认值”的行为不变，避免未请求的 `analyser`/`visualizer` 被运行。

5. **第一阶段验收标准**
   - `pytest tests/qphase -v` 全部通过（新增 smoke 与插件抽取测试）。
   - `qphase run cpu_smoke_kerr_3pa` 在无 GPU 机器上能在 30 秒内完成。
   - `mkdocs build --strict` 无警告。
   - 全仓库 `grep -R "vdp_two_mode\|kerr_cavity"` 不再命中跟踪文件（`.gitignore` 除外）。

---

### 第二阶段：后处理健壮性 + PSD/Lorentz 边界收敛 + 文档字段补全

**目标**：让 `qphase postprocess` 可配置、可审计，让寻峰/拟合有明确测试与序列化规范。

1. **后处理 CLI 增加可审计模式**
   - 在 `packages/qphase/qphase/commands/postprocess.py` 与 `qphase_sde.postprocess` 中新增：
     - `--dry-run` / `--inspect`：只打印将处理的 `.npz` 文件、扫描参数值、mode、输出路径，不写文件。
     - `--freq-min` / `--freq-max`：限制 Lorentz 拟合频率窗口（优先级高于 `--fit-window` 的峰值半宽）。
   - 新增 `tests/qphase_sde/test_postprocess.py` 用例覆盖 `--dry-run` 与频率裁剪。

2. **Lorentz 拟合质量判据**
   - 在 `qphase_sde/postprocess.py::LorentzFitResult` 与 `fit_lorentzian` 中新增：
     - 可配置阈值 `min_r2`、`min_peak_height`、`max_linewidth`；
     - 不满足时仍保留行，但 `status` 设为 `low_quality`（当前只有 `ok`/`failed`）。
   - CLI 暴露 `--min-r2`、`--min-peak-height`、`--max-linewidth`。
   - 新增合成数据测试：低 R²、峰高过低、线宽过大时 `status=low_quality`。

3. **输出 bundle 的 schema/version**
   - 在 `dist_merged.npz` 中写入 `__schema_version__` 与 `__created_by__` 字段。
   - 在 `pdist_merged.pkl` 的 bundle dict 中加入 `__schema_version__`。
   - 文档 `docs/user_guide/output.md(.zh)` 说明这些字段与兼容性策略。

4. **PSD 寻峰测试与序列化**
   - 新增 `tests/qphase_sde/test_peak_finding.py`：
     - 对 `ScipyPeakFinder` 和 `RationalPeakFinder` 分别覆盖无峰、单峰、多峰、拟合失败回退路径。
     - 验证 `PeakInfo.model_dump()` 后的 `properties` 中不含裸 ndarray/complex；若含，则在保存前转换为 list/float。
   - 在 `PsdAnalyzer.analyze()` 保存 peaks 前统一转换 `PeakInfo.properties` 中的 numpy/complex 类型。
   - 为 `kerr_3pa`、`kerr_3mode`、`vdp_level3` 各增加一个轻量导入/实例化测试（计划项 5 剩余部分），可放在 `tests/qphase_sde/test_models.py`。

5. **文档字段与 schema 补全**
   - 更新 `docs/user_guide/output.md(.zh)`：
     - 补全 `.npz` 顶层字段、`analysis["psd"]`、`analysis["dist"]`、`analysis["pdist"]` 的详细结构；
     - 补全 `fit_results.csv` 首列 `job_name`；
     - 说明 `dist_merged.npz` / `pdist_merged.pkl` 字段与 schema/version。
   - 更新 `docs/api/cli.md(.zh)`：
     - 修正 `--version` 不存在的问题；
     - 修正 `qphase template --output / -o` 为仅 `--output`；
     - 加入 `postprocess` 新增参数说明。
   - 每次修改 CLI/输出后运行 `mkdocs build --strict`。

6. **第二阶段验收标准**
   - `pytest tests/qphase_sde -v` 全部通过，新增寻峰、拟合质量、postprocess 测试。
   - `qphase postprocess <run> --scan-param epsilon --dry-run` 只打印不写入。
   - 低质量拟合的 `fit_results.csv` 中 `status=low_quality`。
   - `mkdocs build --strict` 无警告。

---

## 三、推荐优先级

建议按两阶段顺序执行：

1. **先做第一阶段**——它消除当前最明显的用户误导（文档/CLI 命令错误、旧模型名）并建立 CPU smoke，保证后续改动有快速回归。
2. **再做第二阶段**——在稳定基础上增加后处理可选能力与测试深度。

如果人力资源有限，可进一步拆分为：

- **本周最小可用**：完成第一阶段 1、2、3（文档/CLI 清理 + CPU smoke），以及第二阶段 5 的文档字段修正。
- **下周补强**：Scheduler 验证一致性、后处理 dry-run/质量判据、寻峰测试。

---

## 四、依赖与风险

- CPU smoke 依赖 `qphase-sde` 与 `numpy`/`scipy` 已安装；当前 `.venv` 已存在，风险低。
- 修改 `Scheduler._validate_job_dependencies()` 时要避免过度回退全局默认值，导致可选插件被静默注入。
- 为 `PeakInfo.properties` 添加序列化转换时，注意保持 `rational` finder 的 `fitted_curve`/`denominator_zeros` 在后续可视化中仍可用；可考虑保存原始对象同时提供 `model_dump()` 友好版本。
- 新增 CLI 参数需同步中英文文档；`mkdocs-static-i18n` 的 suffix 机制要求 `.zh.md` 与英文文件一一对应。

---

## 五、关键文件速查

- CLI run / help: `packages/qphase/qphase/commands/run.py`
- CLI postprocess: `packages/qphase/qphase/commands/postprocess.py`
- Scheduler validation/run: `packages/qphase/qphase/core/scheduler.py`
- Config loader / plugin extraction: `packages/qphase/qphase/core/config_loader.py`
- Job expansion: `packages/qphase/qphase/core/job_expansion.py`
- Models / registry: `models/`, `models/.qphase_plugins.yaml`, `models/__init__.py`
- PSD / peak finding: `packages/qphase_sde/qphase_sde/analyser/psd.py`, `.../peak_finding/`
- Postprocess core: `packages/qphase_sde/qphase_sde/postprocess.py`
- Docs: `docs/user_guide/*.md(.zh)`, `docs/api/cli.md(.zh)`, `docs/dev_guide/registry.md(.zh)`
- Tests: `tests/qphase/`, `tests/qphase_sde/`
