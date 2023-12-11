# 判别分析

***
## 简介

判别分析（Discriminant Analysis）是最早的统计分类器，于 1936 年由 R. A. Fisher 提出。LDA 在模式识别（如人脸识别等图像识别领域）有着广泛应用。LDA 的核心思想：最大化类间均值，最小化类内方差。

判别分析的主要术语：

- **协方差**（covariance），衡量一个变量与另一个变量协同变化的程度
- **判别函数**（discriminant function），应用于预测变量的函数，能最大限度地分离不同类
- **判别权重**（discriminant weights），应用判别函数得到的打分，用于估计属于不同类别的概率

虽然判别分析包含多种技术，但线性判别分析（linear discriminant analysis, LDA）最常用。Fisher 最初提出的方法与 LDA 略有不同，但机制一致。随着许多更复杂的技术的出现（如逻辑回归、树模型等），LDA 现在应用没那么广泛。

不过在某些程序中依然可能遇到 LDA，，而且 LDA 可以作为其它技术的基础，如 PCA。

!!! attention
    Latent Dirichlet Allocation 的缩写也是 LDA，但是 Latent Dirichlet Allocation 用于文本和自然语言处理，与线性判别分析无关。

## 协方差矩阵

要理解判别分析，首先必须理解两个或多个变量之间的协方差概念。协方差衡量两个变量 $x$ 和 $z$ 之间的关系，假设它们的均值以为 $\overline{x}$  和 $\overline{z}$，那么 x 和 z 的协方差 $s_{x,z}$ 为：

$$s_{x,z}=\frac{\sum^{n}_{i=1}(x_i-\overline{x})(z_i-\overline{z})}{n-1}$$
其中 n 为样本对数，$n-1$ 表示自由度。

和相关系数一样，正数表示正相关，负数表示负相关。但是相关系数在 -1 到 1 之间，而协方差的范围取决于变量 x 和 z 的范围。

x 和 z 的协方差矩阵 $\sum$ 对角线为单个变量方差 $s^2_x$ 和 $s^2_z$，在非对角线为协方差：

$$\widehat{\sum}=\begin{bmatrix}
s_x^2 & s_{x,z} \\
s_{z,x} & x^2_z
\end{bmatrix}$$

```ad-tip
标准差用于将变量归一化为 z-score，而协方差矩阵是该标准化过程的多元（multivariate）扩展。
```

## Fisher 的线性判别

先看一个分类问题，假设我们想用两个连续变量 $(x,z)$ 预测 binary 输出 $y$。从技术上将，判别分析假设预测变量是服从正态分布的连续变量，但是，在实践中，该方法对不是极端偏离正态以及二元预测变量也能很好工作。

Fisher 的线性判别一方面区分 groups 之间的 variation，另一方面还区分 group 内部的 variation。具体来说，为了将数据分为两个 group，LDA 侧重于最大化组间的 $SS_{between}$ 的平方和。对本例，两组对应 $y=0$ 时的 $(x_0, z_0)$ 和 $y=1$ 时的 $(x_1,z_1)$。该方法找到线性组合 $w_xx+w_zz$ 最大化下面的比例：

$$\frac{SS_{between}}{SS_{within}}$$
其中，$SS_{between}$ 是两组组间均值的协方差矩阵加权平方和，$SS_{within}$ 是组内均值的协方差矩阵加权平方和。




