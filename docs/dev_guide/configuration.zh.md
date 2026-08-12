---
description: 配置契约与所有权
---

# 配置系统

配置被拆成四个严格契约：

1. `ProjectManifest` (`qphase.project/2`) 拥有身份与相对路径；
2. `WorkflowSpec` (`qphase.workflow/2`) 拥有元数据和逻辑 Job；
3. `JobConfig` 拥有一次 Engine 调用、插件、scan、input 和保存意图；
4. `SystemConfig` 拥有与 Project 无关的机器/运行策略。

Project 插件默认值位于 `ProjectContext.defaults_path`。它只补全已选择插件的缺失字段，
不会自动激活可选插件命名空间。

## 加载流程

1. `ProjectContext.discover()` 解析唯一 Project；
2. `WorkflowCatalog` 递归索引元数据并保证 ID 唯一；
3. `load_workflow()` 拒绝旧文档并校验严格 wrapper；
4. 顶层插件命名空间被提取到 `JobConfig.plugins`；
5. 合并 Project 默认值与显式 Job 配置；
6. registry schema 校验每个已选择插件；
7. scheduler 校验 Engine manifest 与 Job 图。

Workflow 禁止未知字段。Job 的 `extra="allow"` 只用于动态插件命名空间；未知的非插件
字段不得被解释为新 core 行为。

只有显式 `ScanSpec` 才产生 scan。插件 schema 中的列表始终是科学字面值。编译后得到
轴顺序稳定的不可变 `ParameterGrid`；scheduler 只向 Engine 传递一次，不展开参数点。

`SystemConfigStore` 合并包默认值、站点策略、稀疏用户覆盖、环境指定和显式策略。
读取不创建文件，写入只保存相对包/站点默认值发生变化的字段。Project 路径属于 schema
错误。测试或客户端若需要不同 Session 根目录，必须创建或注入不同 `ProjectContext`。
