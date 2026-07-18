---
description: 配置系统
---

# 配置系统

**配置系统**负责解析、验证和合并仿真参数。它采用分层加载策略，并利用 **Pydantic** 进行严格的模式验证。

## 配置层次结构

系统通过从三个不同层合并配置数据来构建最终执行上下文，按优先级从低到高：

1.  **系统默认值**：包和插件定义中的硬编码默认值。
2.  **全局配置** (`configs/global.yaml`)：用户定义的项目范围设置（例如默认后端、日志详细程度）。
3.  **任务配置** (`configs/jobs/*.yaml`)：实验特定参数。

## 加载管道

配置加载过程遵循严格的管道：

1.  **文件 I/O**：读取 YAML 文件并解析为原始 Python 字典。
2.  **结构规范化**：规范化原始字典以确保结构一致（例如处理简写符号）。
3.  **插件提取**：系统识别与已注册插件命名空间对应的键（例如 `backend`、`model`）。
4.  **模式验证**：
    *   根据 `JobConfig` 模型验证核心任务结构。
    *   根据插件类定义的相应 `config_schema` 验证每个插件配置块。
5.  **合并**：将全局默认值合并到任务配置中，填充缺失的可选字段。

## 使用 Pydantic 进行模式验证

QPhase 使用 Pydantic v2 来强制类型安全和数据完整性。

### `JobConfig` 模型

`JobConfig` 模型定义了仿真任务的结构骨架。

```python
class JobConfig(BaseModel):
    name: str
    engine: dict[str, Any]
    plugins: dict[str, dict[str, Any]]
    params: dict[str, Any]
    scan: ScanSpec | None
    input: InputSpec | None
    system: SystemConfig | None
    # ...
```

### 插件模式

每个插件必须定义一个指向 Pydantic 模型的 `config_schema` 类变量。这允许注册表在插件实例化*之前*验证插件特定参数。

**示例：**
```python
class KerrCavityConfig(BaseModel):
    chi: float = Field(..., gt=0, description="Nonlinearity")
    detuning: float = Field(0.0, description="Frequency detuning")
```

如果用户为 `chi` 提供字符串或负值，Pydantic 验证器将在加载阶段引发描述性错误，防止仿真循环深处的运行时失败。

## 参数扫描 Schema

参数扫描只使用 core 的 `ScanSpec` 表示。它包含按声明顺序排列的命名
`ScanAxisSpec`；每条 axis 包含插件目标路径，以及 `values`、`linspace`、
`logspace` 三者之一。schema 编译后得到不可变的运行时 `ParameterGrid`。

插件 schema 继续描述标量或结构化插件值。core 不再使用 `scanable` 元数据或检查
任意列表来建立 scan；旧 metadata 只能作为识别旧语法并报告迁移错误的标记。
这样可以消除“物理向量”和“参数点集合”之间的歧义。

加载器会对已删除的工作流语法给出可执行的迁移错误，包括：

- 已知标量模型字段使用列表扫描；
- 字符串形式的 `input`；
- `aggregate_input`；
- 本应属于 `SystemConfig` 的 job 顶层运行策略字段。

## 系统运行 Schema

core 的存储、checkpoint、资源提示、进度和 CLI 行为都属于 `SystemConfig`。
内置与用户级默认值由 system config loader 合并；job 可通过现有 `system` 字段覆盖
同一 schema。解析后的值通过 `ExecutionContext` 传给资源包。
