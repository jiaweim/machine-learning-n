# 样条线平滑

## 简介

样条线（Spline）是一种在数学、计算机和工程设计中广泛应用的**平滑曲线**，用于在给定的一组控制点之间插值或逼近一条平滑的路径。它的名称来源于早期造船和制图中使用的弹性木条（称为 样条），这些木条被固定在某些点（knot）上，靠自身的弹性形成一条最平滑的曲线。

在数学上，样条线通常由分段多项式函数构成，每一段都是一个低阶多项式（如二次、三次），但在连接点（节点，knot）处保持一定的连续性。

最常见的类型是**三次样条**（cubic spline），即每段都是三次多项式，并且整体具有 $C^2$ 连续性，即位置、一阶导数和二阶导数都连续。

### 样条类型

1. **插值样条（Interpolating Spline）**

曲线经过所有指定的控制点。例如：自然三次样条。

2. **逼近样条（Approximating Spline）**

曲线不一定经过控制点，而是受气“吸引”而形成平滑形状。例如：B 样条（B Spline）、贝塞尔曲线（Bezier Curve）、NURBS。

### 样条平滑

样条平滑（Spline Smoothing）是一种在**保持数据整体趋势**的同时**去除噪声**或局部波动的数据处理技术。

核心思想：在拟合数据和曲线平滑度之间取得一个平衡。

三次样条因间距



## 参考

- https://bookdown.org/tpinto_home/Beyond-Linearity/smoothing-splines.html
- https://en.wikipedia.org/wiki/Smoothing_spline