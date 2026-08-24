---
description: 资源包契约
---

# 资源包契约

!!! info "Phase 0 契约 —— qphase 2.0"
    本页 `qphase.resource/1` 契约已经获批，并冻结供 Phase 1 在 `qphase.resources` 中实现。
    生产 discovery 与 execution 路径仍只通过 qphase 2.0 分阶段迁移修改。

**资源包**是 QPhase 的可管理资产单元：一个 Python 发行包，恰好打包一个执行 engine、
其插件类、数据产品与公开契约。`qphase_sde` 与 `qphase_cam` 都是资源包；`qphase_viz`
可以采用同一契约并只声明更小的 profile。

资源包的权威描述是其 **`ResourcePackageManifest`**，而不是源码树。安装后的 wheel 不保证
保留可遍历的源码布局，因此 scheduler、registry、CLI 与 GUI 必须仅凭 manifest 与 entry
points 枚举包资产。下述目录骨架是开发与审计约定，由 contract test 与开发期 validator
检查——绝不是运行期发现机制。

## 资源 profile

资源包通过可组合的声明式 profile 承担义务，而不是继承单一整体骨架：

| Profile | 在 `base` 之上的必备项 |
| --- | --- |
| `base` | `manifest.py`、`engine.py`、`config.py`、`state.py`、`result.py`、`errors.py`、`contracts/` |
| `compute` | `planning.py`、`runtime/` |
| `simulation` | `model.py` |

profile 可组合：`qphase_sde` 声明 `base + compute + simulation`；可视化包可以只声明
`base`，不会被强迫创建无意义的 `model.py`。

- `__init__.py` 只导出版本、manifest 与稳定公共类型，禁止 eager import concrete plugins。
- `manifest.py` 是 `ResourcePackageManifest` 的唯一声明位置。
- `engine.py` 是 scheduler 面向该资源包的唯一执行入口。
- `config.py` 保存资源包级公共配置与 task profile，不集中吞并 concrete plugin 配置。
- `state.py` 定义资源包运行状态与可恢复状态协议。
- `result.py` 定义 bundle/result adapter 与命名 data products。
- `errors.py` 定义稳定错误码及其到 core error report 的映射。
- `contracts/` 保存资源包特有且公开的协议、quantity 与 capability。
- `planning.py`（`compute`）把 resolved plugins 与输入产品编译为执行计划；
  `runtime/`（`compute`）只放资源包私有的 arena view、batch/tile 与执行辅助。插件不得
  依赖具体的 runtime scheduler。
- `model.py`（`simulation`）是模型协议与 capability 的稳定入口。

标准化可选资产目录——`math/`、`serialization/`、`_native/`——必须在 manifest 中声明
用途与 `public`/`private` 可见性。其他资源特有目录同样必须声明。不允许在根目录继续
增加 `utils.py` 或含义宽泛的 `core/` 等万能模块。

## 插件类目录

每个插件类占用一个根级命名空间目录（如 `integrator/`、`analyser/`、`peak_finder/`）。
目录内至少包含 `__init__.py` 与 `base.py`（公共契约）；复杂 class 可再有 `config.py`、
`result.py` 或 `contracts.py`。

- concrete plugin 只能位于所属插件类目录。
- 父子插件关系由 manifest 的 slot graph 表达，不由目录嵌套表达。
- concrete plugin 不得直接 import 或构造其他目录的 concrete plugin。

## 资源 manifest

`ResourcePackageManifest`（schema `qphase.resource/1`）至少声明：

- `resource_id`、`schema_version`、`package_version`；
- 唯一 engine 引用与声明的 resource profiles；
- 插件类：namespace、protocol、config schema 引用、entry-point namespace；
- 公开 data products、quantities 与 materializers；
- backend/device/optional-dependency capabilities；
- compatibility range 与确定性 asset fingerprint。

manifest **不**复制 `EngineManifest` 或 concrete `PluginManifest`：engine task
requirements 仍归 `EngineManifest`，child slots 与 concrete 配置仍归 `PluginManifest`。
资源 manifest 只保存稳定引用；`ResourcePackageCatalog` 解析三者并交叉校验（每包恰好
一个 engine、entry point 与 namespace 一致、child graph 无环、task profile 可解析、公共
schema 可导入）。Project、session 与 artifact manifest 保持独立 schema 与命名空间，
不得混用。

## 发现与 overlay

资源包在现有 `qphase` entry-point group 中以 `resource.<id>` 注册其 manifest；core 不
增加平行的 discovery group。`qphase list/show/config` 与未来 GUI 只消费 catalog。

Project-local 与第三方 concrete plugins 作为 **catalog overlay** 注册到资源 manifest
声明的插件类；它们不写回 manifest，也不要求采用完整资源包骨架。overlay provenance
（`package`、`project_overlay`、`third_party`）、compatibility 与 resolved-job
fingerprint 会被记录，使 resolved job snapshot 能区分 package-owned 资产与 overlay。

## 资产 fingerprint

资源资产 fingerprint 由规范化 manifest、package version 与 entry-point descriptors
导出，禁止依赖源码绝对路径、文件 mtime 或目录遍历顺序，确保同一发行包总是产生相同
fingerprint。

## 开发期校验

`qphase.resources.validation` 提供供 contract test 与发布检查使用的 validator：

- 源码布局校验：profile 必备模块存在、concrete plugin 位于其声明的插件类目录、可选
  资产目录已声明；
- manifest 校验：schema 合规、唯一 engine、fingerprint 稳定；
- entry-point 校验按 ownership 划分作用域：`partition_entry_points` 先按 distribution
  把全局 `qphase` group 划分为 package-owned、project-overlay 与 third-party 描述符；
  `validate_package_entry_points` 随后只校验本包 distribution 自有的描述符（恰好一个
  `engine.*`、恰好一个 `resource.<id>`、仅限已声明 namespace），因此共存安装的
  SDE/CAM 与 backend 插件不会触发误报的 engine-count 或 unknown-namespace；
  `validate_overlay_entry_points` 对 project overlay 施加独立、更窄的策略——overlay 按
  namespace 归属到各资源包，且绝不得占用保留的 `resource.*`/`engine.*` namespace。
  third-party 描述符经 `classify_origin` 标注 provenance，并只对各自发行包的 manifest
  校验。

运行期与 wheel 安装只以 manifest 和 entry points 为事实来源；运行期绝不遍历源码树。
