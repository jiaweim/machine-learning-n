# 概率分布

2023-05-24
***

## 分布

- [离散概率分布](./discrete_distribution.md)

- [均匀分布](uniform-distribution.md)
- [伯努利分布](bernoulli.md)
- [二项分布](binomial.md)
- [泊松分布](poisson.md)

连续分布

- [正态分布](./normal.md)

## 数据类型

数据可以分为连续型和离散型。

离散型随机变量只能取有限个值。

连续型随机变量全部可能取值是无穷多的。

## 概念

### 概率密度函数

概率密度函数（Probability Density Function, PDF）对连续型随机变量定义的，本身不是概率，对概率密度函数在某区间内进行积分才得到概率。

$F_X(x)=\int_{-\infty}^xf_X(t)dt$

### 概率质量函数

概率质量函数（Probability Mass Function, PMF）对离散型随机变量，代表值对应的概率。设 $X$ 为离散型随机变量，其全部可能值为 $\{a_1,a_2...\}$。则将：

$$p_i=P(X=a_i), i=1,2,...$$

称为 $X$ 的**概率函数**，或者**概率质量函数**，又称为随机变量 $X$ 的**概率分布**。

概率分布能以列表的形式给出：

|可能值|概率|
|---|---|
|$a_1$|$p_1$|
|$a_2$|$p_2$|
|$\vdots$|$\vdots$|
|$a_i$|$p_i$|
|$\vdots$|$\vdots$|

### 累积分布函数

累积分布函数（Cumulative Distribution Function, CDF）也称为累积密度函数（Cumulative Density Function）或分布函数，为随机变量 `X` 取值小于或等于 `x` 的概率。

CDF 和 PDF 的关系为：

$D(x)=P(X\leq x)=\int_{-\infty}^xP(\xi)d\xi$

所以PDF 是 CDF 的倒数：

$P(x)=D'(x)$

类似的，累积分布函数和离散概率密度函数（概率质量函数）的关系是：

$D(x)=P(X\le x)=\sum_{X\leq x}P(x)$

对任意随机变量 X，其分布函数 F(x) 具有如下性质：

1. `F(x)` 是单调下降的，当 $x_1<x_2$时，有$F(x_1)\leq F(x_2)$
2. 当 $x\rightarrow \infin$时，$F(x) \rightarrow 1$；当 $x \rightarrow -\infin$，$F(x)\rightarrow0$。

### 离散分布

变量只能取离散值的统计分布称为离散分布(discrete distribution)。离散分布的分布函数(distribution function) 为：

$D(x_n)=\sum_{k=1}^nP(x_k)$

其中 $P(x_k)$ 为概率函数。

其总体均值：

$\mu=\frac{1}{N}\sum_{k=1}^Nx_kP(x_k)$

## References

- [List of probability distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions)
- [Statistical Distributions](http://mathworld.wolfram.com/topics/StatisticalDistributions.html)
- [Discrete Distribution](https://mathworld.wolfram.com/DiscreteDistribution.html)
