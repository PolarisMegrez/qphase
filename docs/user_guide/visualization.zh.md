---
description: 用 qphase_viz 渲染 typed product
---

# 可视化

`qphase_viz` 把**类型化结果 bundle 变成图**。它是翻译层而不是分析工具:plotter 通过
`DatasetResultProtocol` 契约选取具名 typed product,决定哪个数组成为哪个轴、线或颜色,然后渲染。
它不运行 analyser、不按路径读文件,也不依赖 `qphase_sde` 或 `qphase_cam`。

同一套 plotter 与配置层也规划为 Phase 11 GUI 预览的后端(当前尚不可用)。出版图仍是建立在它之上的
普通脚本,这也是高度定制版面的做法:用 plotter 保证"数据到图形"的约定,再覆盖版面。

## 运行 engine

一次可视化任务就是一个普通的 Workflow v2 job,消费上游 dataset:

```yaml
schema: qphase.workflow/2
id: typed_viz
title: 类型化可视化
jobs:
  - name: plot_phase
    input: {from: my_simulation, mode: dataset}
    engine:
      viz: {output_dir: figures, format: png}
    visualizer:
      phase_plane:
        plots: [{channel_x: 0, mode: hist2d}]
      power_spectrum:
        plots: [{channels: [0], scale: log}]
```

`output_dir` 与 `format` 是唯一的 engine 级输出设置;每个图的 spec 放在对应的
`visualizer.<插件>` 下。Python 中直接构造同样的对象:

```python
from qphase_viz import VizEngine, VizEngineConfig
from qphase_viz.plotters.phase import PhasePlanePlotter

engine = VizEngine(
    config=VizEngineConfig(output_dir="figures", format="png"),
    plugins={"visualizer": PhasePlanePlotter(plots=[{"channel_x": 0}])},
)
result = engine.run(bundle)  # bundle 实现 DatasetResultProtocol
```

`result.data` 是写出的文件路径列表,`result.metadata` 只记录数量。结果里没有分析载荷,也没有第二份
序列化:一次运行把图持久化在 `output_dir` 下,`result.save(path)` 只是 core `ResultProtocol`
要求的方法,是显式的 no-op。图件是否改为 Artifact attachment 属后续评审项,上述用法不依赖它。

## 样式

图使用具名样式渲染:

| 样式 | 排版 |
| --- | --- |
| `publication`(默认) | STIX 文字与数学,正文 12 pt,轴标签 14.4 pt |
| `sm` | 同样的字体族,整体小一档(正文 10.5 pt) |
| `sans` | Arial + sans 数学,标签 18 pt、刻度 15 pt |

三者共用轴线几何(1 pt 轴线、去掉上/右 spine、刻度朝内、无 legend 边框、300 dpi 导出)。

样式与文字导出可以在 engine 上设一次,也可以逐图覆盖:

```yaml
engine:
  viz:
    output_dir: figures
    format: svg
    style: publication        # 或 sm / sans
    editable: true
    style_overrides: {axes.grid: false}
visualizer:
  time_series:
    plots:
      - channels: [0]
        style: sm                        # 仅这张图
        style_overrides: {axes.labelsize: 12}
```

每张图的优先级是 **单图 > engine > publication 默认**,`style_overrides` 最后应用,
因此显式 rc 值会压过样式表与可编辑开关。

样式作用域是局部 Matplotlib rc 上下文:渲染不会改变进程级 `rcParams`,plotter 调用之间互不污染。

## 可编辑文字

默认 `editable: true`:SVG 保留真实文字(`svg.fonttype: none`),PDF 内嵌 TrueType
(`pdf.fonttype: 42`),标签可在 Inkscape/Illustrator 中直接编辑。`editable: false` 改为输出轮廓。
PNG 不受影响。

字体、字号、标签与标注都是配置而非代码:`style` 决定字号体系,`style_overrides` 调整单个 rc 值,
`BasePlotterConfig` 字段覆盖 `title`、`xlabel`、`ylabel`、`xlim`、`ylim`、`grid`、`figsize`、
`dpi`、`filename`、`constrained_layout` 与 `annotations`。

```yaml
annotations:
  - text: "$\\gamma/\\kappa = 0.1$"
    x: 0.02
    y: 0.95
    coords: axes fraction   # 或 "data"
    fontsize: 9
    color: "#111827"
```

## Plotter

| Plotter | 画什么 | Typed 输入 |
| --- | --- | --- |
| `time_series` | 振幅随时间的演化 | `time_series` / `field_amplitude` product |
| `phase_plane` | Re–Im 或通道–通道平面 | 同一个轨迹 product |
| `power_spectrum` | 单条 PSD 曲线 | `spectral` / `power_spectral_density`(`axis`、`psd`) |
| `parameter_evolution` | 指标随扫描坐标的变化 | typed peak 或轨迹 product |
| `spectral_scan` | 每个 mode 一个 panel 的 PSD 扫描 | 带 scan 轴的 PSD product |
| `heatmap` | 二维标量网格 | 任意含 `x`、`y`、`value[row, column]` 的 product |
| `distribution` | 逐 realization 样本的密度 | 逐 realization 样本 product |

单点 plotter(`time_series`、`phase_plane`、`power_spectrum`)拒绝多点扫描:请把输入按点 map,
或显式选一个点,不要依赖隐式平均。

### PSD 扫描

`spectral_scan` 每个 mode 一个 panel,共享频率轴,每个选中扫描点一条曲线,图例由扫描坐标生成。
扫描坐标必须显式命名:

```yaml
visualizer:
  spectral_scan:
    plots:
      - scan_parameter: omega        # PSD product 上存的扫描坐标
        channels: [0, 1, 2]          # 每个 mode 一个 panel
        scan_indices: [0, 10, 20]    # 或用 scan_coordinates,见下
        log_scale: true
        label_template: "{name}={value:.6g}"
```

按坐标值选点必须给出显式容差:`scan_coordinates: [0.12, 0.24]` **同时**给
`scan_tolerance: 0.01`。没有默认容差——无界最近邻会静默画出与请求不同的扫描点;超出容差即报错。
曲线的颜色来自它在整个扫描中的位置,因此画子集不会重排其它曲线。

### 标量热图

`heatmap` 渲染一个 `pcolormesh` 加 colorbar。`x` 是 1-D row 坐标、`y` 是 1-D column 坐标、
`value[row, column]` 是网格,声明方向与渲染结果因此一致:

```yaml
visualizer:
  heatmap:
    plots:
      - x_variable: gamma_b
        y_variable: gamma_c
        value_variable: local_information
        valid_variable: physical       # 可选布尔掩码,False 留白
        cmap: flow                     # "scan"、"flow" 或任意 Matplotlib 色图
        norm: linear                   # 或 log(需严格正值)
        colorbar_label: Information
        contours:
          - variable: continuation_allowed
            levels: [0.5]
        series:
          - x: [0.8]
            y: [0.957]
            marker: "*"
            linestyle: null
```

非有限单元与掩码单元都留白,**不会当作 0**。网格形状与坐标不符、没有有限有效值、log 档遇到非正值
都会明确报错;product 选择歧义时要求指定 `product`。

### 样本分布

`distribution` 画上游已算好的样本分布。`value_variable` 存 `value[realization]` 或
`value[scan, realization]`;选择多个扫描点会画多条曲线,**绝不合并样本**。

```yaml
visualizer:
  distribution:
    plots:
      - value_variable: estimate
        scan_parameter: omega
        scan_indices: [0, 10]        # 或 scan_coordinates + scan_tolerance
        center: reference_mean       # 首个 series 的均值映射到 0
        scale: -1                    # reference - value
        bin_width: 0.0001            # 或 "bins: 40"
        labels: [reference, perturbed]
        show_means: true
        show_fwhm: true
        shift_label: signal shift    # 需 show_means 且两条 series
        width_label: noise width     # 需 show_fwhm
```

`center` 决定原点、`scale` 决定符号:`center: none` 画 `scale * value`,
`center: reference_mean` 画 `scale * (value - reference)`,reference 取首个选中 series 的均值。
密度按 `counts / (N * bin_width)` 归一化,`N` 是**全部有限样本数**:被 `xlim` 裁剪的窗口总面积
小于 1,而曲线高度仍是完整分布的密度。半高宽要求峰完整落在范围内,否则报错而不是从数组边界外插。
