# Declarative 数据可视化

## 简介

除了采用命令式 API 进行数据可视化，smile 还支持声明式数据可视化。

`smile.plot.vega` 包创建一个规范，将可视化描述为从数据到图形属性的映射。该规范基于 [Vega-Lite](https://vega.github.io/vega-lite/)。Vega-Lite 编译器自动生成可视化组件，包括坐标轴、图例和 scale 等。然后，它根据一组规则确定这些组件的属性。

这种方法使得规范简洁易懂，同时又便于设置。由于 Vega-Lite 专门为分析而设计，它支持数据转换，如 aggregation, binning, filtering, sorting 以及可视化转换，如 stacking 和 faceting。此外，Vega-Lite 规范可以组合多个 layer 和 multi-view 显示，以及交互。

Vega-Lite 网站提供了该规范的详细文档，下面将通过示例展示如何创建各种 charts。

## Bar Charts

### Simple Bar Chart

