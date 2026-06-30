## Plan: QPhase Service、GUI、文档与 Postprocess 架构实施计划

### 0. 总目标

把 QPhase 重构为更清晰、可复用、可扩展的分层架构：

- `qphase` core 提供公共能力：backend、registry、scheduler、config、service API、CLI facade、result schema。
- CLI 与 GUI 是平级客户端，都调用 service layer，不互相包装。
- service layer 至少包含三条主线：scheduler service、registry service、config service。
- config pipeline 明确区分 system/global/job/plugin schema/engine manifest 的所有权、生命周期和合并顺序。
- 文档面向两类用户：物理研究者用户与专业开发者。
- GUI 初期轻量化，突出 CLI 不擅长的图形化操作、配置可视化、执行计划、进度追踪、结果浏览。
- postprocess 初期不作为独立 package，也不作为单一 plugin；应作为 `qphase_sde` 内的 package-level workflow engine/use-case，并逐步拆出 `aggregator`、`fitter`、`exporter` category plugins。

### 1. 关键设计原则

1. CLI 与 GUI 平级
   - CLI 只负责命令行输入/输出、参数解析、脚本友好体验。
   - GUI 只负责可视化交互、配置展示、执行计划和结果浏览。
   - 两者都调用 `qphase` service layer。
   - 不允许 GUI 调 Typer command 函数，也不建议 CLI 绕过 service 直接调底层实现。

2. Service layer 是 core 的应用 API
   - scheduler service: job/workflow/run/progress/artifact。
   - registry service: plugin discovery/catalog/schema/validation/manifest。
   - config service: system/global/job load-normalize-merge-validate-snapshot。
   - service layer 返回结构化对象，不打印 Rich/Typer 输出。

3. Config pipeline 必须显式
   - system config: core runtime behavior/path/defaults。
   - global config: 用户设备/环境偏好，例如 backend、dtype、cache、默认输出目录。
   - job config: 单次任务意图，例如 engine、model 参数、scan、input/output、显式启用 analyser/visualizer/postprocess。
   - plugin schema: plugin/package 拥有的配置校验契约。
   - engine manifest: 决定 job 运行时需要哪些 plugin namespace，也决定 global default 是否可继承。
   - global config 不应隐藏 workflow step；optional plugin 不应仅因 global 默认存在就自动运行。

4. 文档双入口
   - 用户文档：物理研究者可运行、可改参数、可看结果、可复现。
   - 开发者文档：插件作者/维护者理解 architecture、service、registry、scheduler、config、result schema、dependency rules。

5. Postprocess 的归属
   - 当前 postprocess 强依赖 SDE result、PSD payload、distribution、scan metadata，因此初期放在 `qphase_sde`。
   - 形式上是 `engine.sde_postprocess` 或 package-level workflow/use-case。
   - 内部拆分：single-result analysis 仍归 `analyser`；cross-result merging 归 `aggregator`；拟合归 `fitter`；文件导出归 `exporter`。
   - 只有当 postprocess 完全依赖 core result schema、可跨领域复用时，才考虑抽成 `qphase_postprocess` 或 `qphase_analysis`。

### 2. 当前代码锚点

- `packages/qphase/qphase/core/scheduler.py`
  - 已有 `Scheduler`、`JobProgressUpdate`、`JobResult`、`run(job_list, dry_run=False, resume_from=None)`。
  - 已有 progress callback 基础。
  - 需要结构化 execution plan API 和 service facade。

- `packages/qphase/qphase/core/config.py`
  - `JobConfig`、`JobList` 是 workflow schema 中心。

- `packages/qphase/qphase/core/config_loader.py`
  - `load_jobs_from_files()`、`list_available_jobs()`、`get_config_for_job()`、`_extract_plugin_fields()`。
  - 需要明确 config service 封装，避免 CLI/GUI 重复处理 config。

- `packages/qphase/qphase/core/registry.py`
  - plugin discovery/list/schema/validate/create instance。
  - GUI schema 表单和 developer docs 的数据来源。

- `packages/qphase/qphase/core/protocols.py`
  - `EngineManifest`、`ResultProtocol`、`EngineBase`。
  - 后续可新增 result schema/exporter 相关协议。

- `packages/qphase/qphase/commands/run.py`
  - 当前 CLI orchestration 较多，应逐步改为 service layer client。

- `packages/qphase/qphase/commands/postprocess.py`
  - 当前 core 中有 SDE-specific postprocess facade，长期应改为 service/plugin facade 或迁出 core 语义。

- `packages/qphase_sde/qphase_sde/postprocess.py`
  - 当前 postprocess 功能组合体：load、aggregate、fit、export。
  - 应拆职责，并由 `qphase_sde` postprocess engine/use-case 组合调用。

- `packages/qphase_sde/qphase_sde/result.py`
  - 当前 `.npz` save/load 使用 object/pickle 风格。
  - 后续需 result manifest/sidecar、外部友好格式。

- `packages/qphase_viz/qphase_viz/engine.py`
  - `VizEngine` 可复用生成静态图像。
  - 需要添加 `EngineManifest`，减少对 SDE 具体类的依赖。

- `packages/qphase_viz/pyproject.toml`
  - 当前 `qphase_viz` mandatory 依赖 `qphase-sde`，长期应改 optional/integration。

### 3. 实施路线总览

按低风险顺序推进：

1. 文档和契约先行，不先大改行为。
2. 新增 service layer facade，先包裹现有能力。
3. 给 scheduler 增加 execution plan 结构化输出。
4. CLI 渐进迁移到 service layer。
5. 轻量 GUI MVP 基于 service layer 做。
6. postprocess 拆分职责并作为 `qphase_sde` workflow engine 进入 scheduler。
7. 输出 schema 与 package 解耦后续推进。

### 4. Phase 1: 文档与架构契约

目标：先把原则写清楚，让后续实现有稳定标准。

#### 4.1 用户文档改造

修改或新增：

- `docs/user_guide/getting_started.md`
- `docs/user_guide/getting_started.zh.md`
- `docs/user_guide/configuration.md`
- `docs/user_guide/configuration.zh.md`
- `docs/user_guide/output.md`
- `docs/user_guide/output.zh.md`
- 新增 `docs/user_guide/gui.md`
- 新增 `docs/user_guide/gui.zh.md`

用户文档应包含：

- QPhase 是什么：面向物理/量子光学/随机微分方程/相空间仿真的工作流工具。
- 最小运行流程：安装、选择 CPU backend、运行一个 demo job、查看输出。
- YAML job 基础：engine、model、backend、integrator、analyser、visualizer。
- 参数扫描基础：列表参数、cartesian、zipped、输出目录。
- Global config 对用户意味着什么：设备默认值、backend、dtype、输出路径。
- Job config 对用户意味着什么：一次具体计算任务。
- 输出解释：session manifest、result files、figures、CSV/JSON、`.npz`。
- CLI 常用命令。
- GUI 用户路径：浏览 job、修改参数、查看 execution plan、开始 run、看进度、打开结果。
- FAQ：CUDA 不可用怎么办、结果在哪里、如何复现实验、如何只改后处理。

写作要求：

- 不要求用户理解 registry internals。
- 尽量使用物理任务例子，而不是软件架构术语开头。
- 示例应能 CPU 运行。
- 中英文同步。

#### 4.2 开发者文档改造

修改或新增：

- `docs/dev_guide/architecture.md`
- `docs/dev_guide/architecture.zh.md`
- `docs/dev_guide/plugin_development.md`
- `docs/dev_guide/plugin_development.zh.md`
- `docs/dev_guide/scheduler.md`
- `docs/dev_guide/scheduler.zh.md`
- `docs/dev_guide/registry.md`
- `docs/dev_guide/registry.zh.md`
- `docs/dev_guide/configuration.md`
- `docs/dev_guide/configuration.zh.md`
- 新增 `docs/dev_guide/service_layer.md`
- 新增 `docs/dev_guide/service_layer.zh.md`
- 新增 `docs/dev_guide/postprocess.md`
- 新增 `docs/dev_guide/postprocess.zh.md`

开发者文档应明确：

- 分层架构：client -> service -> core primitives -> resource package -> category -> plugin。
- 依赖方向：resource package depends on qphase core；sibling package 不应 mandatory 互相依赖。
- CLI/GUI 平级关系。
- scheduler service lifecycle。
- registry service lifecycle。
- config service pipeline。
- package/category/plugin 标准组织结构。
- engine 放 package 级；category 放 protocol/base/config conventions；plugin 是能力单元。
- postprocess 为什么不是 analyser，不是独立 package，当前为什么放 `qphase_sde`。
- 何时可抽象新 category 或新 package。

#### 4.3 文档验证

运行：

- `uv run mkdocs build --strict`

如果当前文档本身已有 warning，需要记录并只修与本次改动相关的问题。

### 5. Phase 2: Service layer 基础

目标：新增 Python-first service facade，先不引入 GUI 技术栈。

建议新增 package/module：

- `packages/qphase/qphase/service/__init__.py`
- `packages/qphase/qphase/service/scheduler.py`
- `packages/qphase/qphase/service/registry.py`
- `packages/qphase/qphase/service/config.py`
- `packages/qphase/qphase/service/models.py`

如果项目偏好 `core/service.py` 也可，但推荐独立 `qphase.service`，表示这是应用层 API。

#### 5.1 Service models

在 `service/models.py` 定义结构化返回模型，可用 dataclass 或 Pydantic。建议 Pydantic，因为 GUI/API 更容易 JSON 化。

建议模型：

- `PluginSummary`
  - `namespace`
  - `name`
  - `package` optional
  - `description`
  - `schema_available`
  - `entry_point` optional

- `PluginCatalog`
  - `packages`
  - `namespaces`
  - `plugins`

- `ConfigSource`
  - `kind`: system/global/job/merged/snapshot
  - `path`
  - `data`

- `ConfigValidationIssue`
  - `level`: error/warning/info
  - `path`
  - `message`
  - `source`

- `MergedConfigPreview`
  - `job_name`
  - `raw_job_config`
  - `global_defaults_used`
  - `merged_config`
  - `validation_issues`

- `ExecutionPlan`
  - `session_preview_id` optional
  - `original_jobs`
  - `expanded_jobs`
  - `edges`
  - `scan_groups`
  - `artifacts`
  - `validation_issues`

- `ExecutionPlanJob`
  - `name`
  - `base_name`
  - `engine`
  - `plugins`
  - `scan_params`
  - `input`
  - `output`
  - `save`
  - `expected_run_subdir`

- `ExecutionPlanEdge`
  - `source`
  - `target`
  - `kind`: input/output/aggregate/depends_on

- `RunHandle`
  - `session_id`
  - `session_dir`
  - `status`

- `ArtifactSummary`
  - `path`
  - `kind`: result/figure/table/manifest/log/other
  - `format`
  - `job_name`
  - `metadata`

#### 5.2 Config service

新增 `ConfigService`，职责：

- `load_system_config()` wrapper。
- `load_global_config(path=None)`。
- `save_global_config(data, path=None)`。
- `load_job_files(paths)` returns `JobList`。
- `normalize_job_config(raw)`。
- `merge_for_job(job, system_config=None)`。
- `preview_merged_config(job)` returns `MergedConfigPreview`。
- `validate_against_registry(job_or_config)`。

重点实现原则：

- 不直接实例化 plugin；只做 merge/validate/preview。
- merge 逻辑应复用现有 `get_config_for_job()` 和 scheduler 中 manifest filtering 的规则，避免重复且分叉。
- 如果现有 merge/filter 逻辑只在 `Scheduler._run_job()` 内，应考虑抽出 helper，例如 `resolve_effective_plugins_config(job, system_config, registry)`。
- optional namespace 继承规则必须测试。

#### 5.3 Registry service

新增 `RegistryService`，职责：

- `discover(include_local=True)`。
- `list_plugins(namespace=None)`。
- `get_catalog()`。
- `get_schema(namespace, name)`。
- `validate_config(namespace, name, config)`。
- `get_engine_manifest(engine_name)`。
- `get_scanable_params(namespace, name)`。

原则：

- 只返回普通 dict/Pydantic 模型，不打印 Rich。
- 捕获并返回插件导入错误/不可用状态，GUI 需要显示。
- 复用现有 registry/discovery。

#### 5.4 Scheduler service

新增 `SchedulerService`，职责：

- `list_jobs()`。
- `load_jobs(names_or_paths)`。
- `build_plan(job_list, system_config=None)` returns `ExecutionPlan`。
- `run(job_list, progress_callback=None, resume_from=None)`。
- `dry_run(job_list)` returns `ExecutionPlan` or dry-run result。
- `list_artifacts(session_dir)`。
- `load_session_manifest(session_dir)`。

实现重点：

- 初期 `run()` 可以直接包裹 `Scheduler.run()`。
- `build_plan()` 应复用 `Scheduler._validate_jobs()` 和 `_expand_parameter_scans()`，但如果这些是私有方法，不要在多个地方复制逻辑；可新增 public-ish helper。
- plan 不应创建真实 run session，或只创建 preview id；避免 dry-run 产生多余目录。如果当前 `Scheduler.run(dry_run=True)` 会 initialize session，可先保守实现，但后续应改为无副作用 plan。

#### 5.5 CLI 渐进迁移

修改：

- `packages/qphase/qphase/commands/run.py`

目标：

- `qphase run jobs` 保持行为不变。
- 内部尽量调用 `SchedulerService`/`ConfigService`。
- 新增或规划 CLI options：
  - `--dry-run`
  - `--plan`
  - `--resume-from`
  - `--json` for machine-readable output

如果一次改动风险高，可先只新增 service，不迁 CLI；后续单独 PR 迁移。

#### 5.6 Phase 2 验证

运行：

- `uv run pytest tests/qphase -q`
- `uv run ruff .`
- 如果 ruff 已知失败，先读取失败原因，只修本次相关问题。
- `uv run pre-commit run --all-files`

新增测试建议：

- `tests/qphase/service/test_registry_service.py`
- `tests/qphase/service/test_config_service.py`
- `tests/qphase/service/test_scheduler_service.py`

### 6. Phase 3: ExecutionPlan

目标：让 CLI/GUI/dry-run 共用一个结构化执行计划。

#### 6.1 ExecutionPlan 内容

应包含：

- 原始 job 列表。
- 参数扫描展开后的 job 列表。
- 每个 expanded job 的 base job、index、scan params。
- input/output/aggregate/depends_on edges。
- engine name。
- required plugins。
- explicit plugins。
- inherited global defaults。
- optional plugins 是否被启用。
- 预期输出目录/文件名。
- validation issues。

#### 6.2 实现位置

优先位置：

- plan models: `packages/qphase/qphase/service/models.py`
- plan builder: `packages/qphase/qphase/service/scheduler.py`

若底层逻辑需要复用，可在 core 增加：

- `packages/qphase/qphase/core/planning.py`

避免直接把大量 plan 构造塞进 CLI。

#### 6.3 GUI/CLI 对 plan 的用法

CLI：

- `qphase run jobs my_job --plan`
- 输出 human-readable table 或 JSON。

GUI：

- execution graph。
- scan 展开预览。
- final merged config preview。
- before-run validation。

#### 6.4 测试

覆盖：

- 单 job 无 scan。
- 单 job cartesian scan。
- zipped scan。
- SDE -> Viz input edge。
- aggregate input。
- optional analyser 不因 global 默认自动运行。

### 7. Phase 4: GUI MVP

目标：轻量实现，只做 CLI 不擅长的东西。

#### 7.1 技术路线选择

推荐优先级：

1. Python FastAPI + local web frontend。
2. VS Code WebView + Python service subprocess。
3. Streamlit/Gradio 快速原型。

建议正式 MVP 用 FastAPI + React/Vite 或 Svelte。Streamlit/Gradio 可用于内部验证，但长期可维护性和 workflow graph/control 不如自建轻前端。

#### 7.2 后端 API

建议 endpoints：

- `GET /health`
- `GET /jobs`
- `GET /jobs/{name}`
- `POST /jobs/validate`
- `POST /plans`
- `GET /plugins`
- `GET /plugins/{namespace}/{name}/schema`
- `GET /config/global`
- `PUT /config/global`
- `POST /runs`
- `GET /runs/{session_id}`
- `GET /runs/{session_id}/events` via SSE/WebSocket
- `GET /runs/{session_id}/artifacts`
- `GET /artifacts?path=...`

#### 7.3 GUI pages

First version pages：

- Jobs page
  - list jobs
  - open YAML
  - run / plan buttons

- Config page
  - global config editor
  - validation status
  - show inherited defaults preview

- Plugins page
  - package/category/plugin tree
  - schema viewer
  - availability diagnostics

- Plan page
  - expanded jobs list
  - simple dependency graph
  - scan parameter table
  - validation warnings

- Run page
  - progress bar
  - current job/stage/message
  - ETA
  - event log

- Results page
  - session manifest
  - generated figures
  - CSV/JSON artifacts
  - open file/folder links

#### 7.4 Explicit exclusions

Do not implement in MVP：

- Full drag-and-drop workflow editor。
- Full interactive plotting engine。
- Remote cluster execution。
- Multi-user auth/server。
- Plugin source-code editor。
- Notebook replacement。

#### 7.5 GUI validation

Use a tiny CPU job：

- plan job。
- run job。
- watch progress。
- display generated figure from `qphase_viz`。
- display output artifacts。

### 8. Phase 5: Postprocess decomposition

目标：把当前 `postprocess.py` 从“一团 use-case 函数”拆成清晰职责。

#### 8.1 目标结构

在 `qphase_sde` 内逐步建立：

- `qphase_sde/postprocess/` or `qphase_sde/workflows/postprocess/`
  - package-level use-case / engine

- `qphase_sde/aggregators/`
  - `scan_psd.py`
  - `distribution.py`

- `qphase_sde/fitters/`
  - `lorentzian.py`

- `qphase_sde/exporters/`
  - `csv_bundle.py`
  - `npz_internal.py`

具体目录可依据现有包风格调整，但 category/plugin 名称应进入 registry。

#### 8.2 Engine 设计

新增 engine：

- entry point: `engine.sde_postprocess = qphase_sde.postprocess.engine:SDEPostprocessEngine`

Engine manifest：

- required maybe: `aggregator`, `exporter`
- optional: `fitter`

注意：如果 `aggregator/fitter/exporter` category 还未被 core registry 文档化，需确保 registry 支持任意 namespace；当前 registry 看起来支持。

#### 8.3 YAML 示例

目标 YAML 概念：

- job 1: SDE scan。
- job 2: postprocess。
  - `input: vdp_sde`
  - `aggregate_input: ...`
  - `engine: sde_postprocess`
  - `aggregator: scan_psd`
  - `fitter: lorentzian`
  - `exporter: csv_bundle`
- job 3: viz optional。

#### 8.4 Backward compatibility

- 保留 `qphase postprocess` CLI。
- 它内部调用 `qphase_sde` postprocess service/engine 或生成等价 temporary JobList。
- 当前 `postprocess_run()` 和 `export_postprocess_bundle()` 可先作为 compatibility wrapper。

#### 8.5 测试

覆盖：

- loading existing `.npz`。
- PSD axis alignment。
- missing scan_param error。
- Lorentz fit success/failure。
- quality threshold status。
- CSV bundle output。
- old CLI command still works。
- YAML postprocess job works。

### 9. Phase 6: Output schema

目标：保留 `.npz`，但建立外部友好格式。

#### 9.1 `.npz` 定位

- `.npz` 是 internal efficient artifact。
- 允许存在，但不是唯一稳定公开格式。
- 避免公开契约依赖 `allow_pickle=True`。

#### 9.2 Result manifest

每个 result/run 输出：

- `result.qphase.json` or session-level artifact manifest。

字段：

- `schema_version`
- `producer_package`
- `producer_version`
- `engine`
- `job_name`
- `run_id`
- `scan_params`
- `config_snapshot`
- `artifacts`
- `arrays`
- `analysis`
- `units/conventions`

#### 9.3 Export formats

First support：

- CSV for tables。
- JSON for metadata/schema。
- NPZ for internal arrays。

Later support：

- HDF5 或 Zarr，视依赖政策决定。

### 10. Phase 7: Package decoupling

目标：恢复并强化原始原则。

#### 10.1 qphase_viz 解耦

- Add `EngineManifest` to `VizEngine`。
- `qphase_viz` visualizers consume standard `ResultProtocol`/result schema。
- Remove mandatory dependency on `qphase-sde` from `packages/qphase_viz/pyproject.toml`。
- If SDE-specific visualizers need SDE internals, move them to optional extra or integration package。

#### 10.2 Core 去领域化

- Core CLI should not own SDE-specific postprocess semantics long term。
- Options:
  - generic plugin command facade。
  - `qphase_sde` provides its own CLI group。
  - core `qphase postprocess` dispatches to installed handlers through registry/service。

### 11. Implementation guardrails

- Keep changes incremental。
- Do not break current YAML configs。
- Do not remove current postprocess CLI immediately。
- Add tests near each new service/category。
- Prefer wrapping existing behavior before refactoring internals。
- Avoid introducing GUI before service APIs are testable without GUI。
- Keep CPU smoke data small。
- Do not make HDF5/Zarr mandatory in early phases。
- Preserve `.npz` compatibility during transition。

### 12. Suggested first implementation PR

If starting implementation now, the best first PR is documentation + service skeleton, not GUI or postprocess rewrite.

Scope：

1. Add developer docs for architecture/service/config/postprocess principles。
2. Add user docs framing for physicist users。
3. Add `qphase.service` package with empty/initial facades:
   - `SchedulerService`
   - `RegistryService`
   - `ConfigService`
   - service models
4. Add minimal tests for registry/config service wrappers。
5. Do not change runtime behavior yet except through well-tested wrappers。

Why first：

- Lowest risk。
- Establishes language and boundaries。
- Lets CLI/GUI/postprocess work converge on same contracts。

### 13. Validation checklist

After each implementation phase run appropriate subset:

- `uv run ruff .`
- `uv run pytest tests/qphase -q`
- `uv run pre-commit run --all-files`
- `uv run mkdocs build --strict`

If `uv run ruff .` fails while `pre-commit` passes, inspect whether ruff command includes generated/site files or docs build artifacts. Avoid fixing unrelated generated outputs unless necessary.

### 14. Definition of done

Short-term done：

- Docs clearly distinguish user vs developer audience。
- CLI and GUI architecture documented as peer clients。
- `qphase.service` skeleton exists and wraps registry/config/scheduler basics。
- ExecutionPlan design documented or partially implemented。
- Postprocess placement decision documented: `qphase_sde` workflow engine, not new package for now。

Medium-term done：

- CLI uses service layer for job run/list/plan。
- GUI MVP can browse jobs, preview plan, run a small job, track progress, browse results。
- Postprocess can run as YAML scheduler job。
- `.npz` has manifest/sidecar and CSV/JSON export companion。

Long-term done：

- `qphase_viz` no longer mandatory-depends on `qphase_sde`。
- Postprocess categories are cleanly registered and documented。
- Result schema is stable enough for external reuse。
