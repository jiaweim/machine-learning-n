# 方差

2022-03-12, 20:52
***

## 定义

设 X 是一个随机变量，若 $E\{[X-E(X)]^2\}$ 存在，则称 $E\{[X-E(X)]^2\}$ 为 X 的方差，记为 D（X）或 Var(X)，即：

$$D(X)=Var(X)=E\{[X-E(X)]^2\} \tag{1}$$

在应用上还引入量 $\sqrt{D(X)}$，记为 $\sigma (X)$，称为**标准差**或均方差。

按照定义，随机变量 X 的方差表达了 X 的取值与其数学期望的偏离程度：

- 若 D(X) 较小，意味着 X 的取值集中在 E(X) 的附近；
- 反之，若 D(X) 较大，意味着 X 的取值较分散。

因此，D(X) 是刻画 X 取值**分散程度**的一个量。

由方差的定义可知，方差实际上是随机变量 X 的函数 $g(X)=(X-E(X))^2$ 的数学期望。于是对于离散型随机变量，有：

$$D(X)=\sum_{k=1}^\infty [x_k-E(X)]^2p_k \tag{2}$$

其中 $P(X=x_k)=p_k, k=1,2,...$ 是 X 的分布律。

对于连续型随机变量，有：

$$D(X)=\int _{-\infty}^{\infty}[x-E(X)]^2f(x)dx \tag{3}$$

其中 f(x) 是 X 的概率密度。

随机变量 X 的方差可按下列公式计算：

$$D(X)=E(X^2)-[E(X)]^2$$