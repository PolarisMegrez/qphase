---
description: GUI 与本地 API
---

# GUI 与本地 API

QPhase 的 GUI 工作从本地 API 后端开始。目标是暴露 CLI 已经使用的同一层 service layer，让未来的浏览器界面可以浏览 job、检查 plugin、预览 execution plan、管理配置，而不需要调用 Typer command 函数。

## 安装 GUI 依赖

本地 API 是可选功能。需要运行或开发 GUI 时，安装 GUI extra：

```bash
pip install "qphase[gui]"
```

如果使用源码 checkout，安装标准开发环境即可：

```bash
uv sync --dev
```

## 启动本地 API

启动本地 GUI 最快的方式是：

```bash
qphase gui
```

该命令会启动 FastAPI 后端，并在 `http://127.0.0.1:8000` 提供内置 Web Console。

开发时，也可以直接使用 FastAPI app factory：

```bash
uvicorn qphase.gui.api:create_app --factory --reload
```

如需修改监听地址，可使用 `qphase gui --host 0.0.0.0 --port 8080`。

## Web Console

内置 console 是轻量浏览器 UI，包含六个视图：

| 视图 | 用途 |
| :--- | :--- |
| Jobs | 列出可用 job，打开 YAML 派生的 job 数据，执行 plan 和 run。 |
| Config | 以 JSON 查看和编辑 global config。 |
| Plugins | 浏览已注册 plugin namespace 和 schema 可用状态。 |
| Plan | 为选中的 job 构建 execution plan。 |
| Run | 启动同步本地 run，并捕获 progress events。 |
| Results | 查看 session manifest、run events、artifacts 和 JSON/text payload。 |

## 可用端点

第一版后端切片聚焦读取和预览 workflow：

| 端点 | 用途 |
| :--- | :--- |
| `GET /health` | 检查本地 API 是否运行。 |
| `GET /jobs` | 从配置目录列出可用 job 名称。 |
| `GET /jobs/{name}` | 按名称加载 job 配置。 |
| `POST /jobs/validate` | 按 registry schema 校验 raw job 对象。 |
| `POST /plans` | 为选中的 job 构建 execution plan。 |
| `POST /runs` | 同步执行选中的 job，并返回 run handle 与结果。 |
| `GET /plugins` | 列出 plugin catalog，可按 namespace 过滤。 |
| `GET /plugins/{namespace}/{name}/schema` | 返回 plugin JSON schema。 |
| `GET /config/global` | 读取 global config。 |
| `PUT /config/global` | 保存 global config。 |
| `GET /runs/{session_id}` | 读取 session manifest。 |
| `GET /runs/{session_id}/events` | 读取当前本地 API 进程记录的 run progress events。 |
| `GET /runs/{session_id}/artifacts` | 列出 session 目录下的 artifacts。 |
| `GET /artifacts?path=...` | 读取 JSON/text artifact payload，或返回二进制文件元数据。 |

plan 请求示例：

```bash
curl -X POST http://127.0.0.1:8000/plans \
  -H "Content-Type: application/json" \
  -d '{"jobs": ["vdp_sde"]}'
```

run 请求示例：

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"jobs": ["vdp_sde"]}'
```

run 完成后，可使用返回的 `session_id` 查看事件和 artifacts：

```bash
curl http://127.0.0.1:8000/runs/SESSION_ID/events
curl http://127.0.0.1:8000/runs/SESSION_ID/artifacts
curl "http://127.0.0.1:8000/artifacts?path=/absolute/path/to/session_manifest.json"
```

## MVP 范围

这个 MVP 刻意保持轻量。它提供本地 Web Console 和 API，用于 job 浏览、plugin 浏览、config 编辑、plan preview、同步本地 run、progress events 和 result artifact 查看。当前不包含拖拽式 workflow 编辑、远程执行、用户账户或 notebook 替代功能。
