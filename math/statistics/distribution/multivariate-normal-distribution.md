# 多元正态分布

## 简介

正如一元正态分布（univariate normal distribution）是单变量统计中最重要的统计分布一样，多元正态分布（multivariate normal distribution）也是多变量分析中最重要的分布。

为什么多变量正态分布如何重要？可能有以下三个原因：

1. 数学的简洁性。该分布相对容易处理，容易获得基于该分布的多元方法。
2. 中心极限定理的多元版本。

## 多元正态分布定义

一元正态分布 $N(\mu,\sigma^2)$ 的概率密度函数为：

$$
\begin{align}
f(x)&=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\
&=(2\pi)^{-1/2}(\sigma^2)^{1/2}exp[-\frac{1}{2}(x-\mu)(\sigma^2)^{-1}(x-\mu)],\quad -\infty < x < \infty
\end{align}
$$


## 参考

- https://brilliant.org/wiki/multivariate-normal-distribution/
- https://online.stat.psu.edu/stat505/book/export/html/636