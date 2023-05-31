# 二元正态分布（Bivariate Normal Distribution）

2023-05-29
***
## 简介

设随机变量 Y 表示随机选择个体的体重（磅）。现在我们想确定一个随机选择的个体体重在 140 到 160 磅之间的概率。即计算 $P(140<Y<160)$？

另外，我们可以想象，人的体重会随着身高的增加而增加，因此，在计算一个随机选择的人体重在 140 磅到 160 磅之间的概率时，首先考虑其身高 X 可能更好。即，计算 $P(140<Y<160|X=x)$。要计算该条件概率，首先要确定条件分布。我们先做一些假设：

1. $Y$ 服从正态分布；
2. 给定 $x$ 时 Y 的条件均值 $E(Y|x)$ 在 x 上是线性的；
3. 给定 $x$ 时 $Y$ 的条件方差 $Var(Y|x)$ 为常数。

基于这三个假设，可以得到给定 $X=x$ 时 Y 的条件分布。

在以上三个假设的基础上，再加上随机变量 X 服从正态分布的假设。基于这四个价格，能确定 X 和 Y 的联合概率密度函数。

主要内容：

- 给定 $X=x$ 时 Y 的条件分布，假设：
	- Y 服从正态分布
	- 给定 x 时 Y 的条件均值 $E(Y|x)$ 在 x 上是线性的
	- 给定 x 时 Y 的条件方差 $Var(Y|x)$ 是常数
- 根据条件分布计算条件概率
- 计算 X 和 Y 的联合分布，假设：
	- X 服从正态分布
	- Y 服从正态分布
	- 给定 x 时 Y 的条件均值 $E(Y|x)$ 在 x 上是线性的
	- 给定 x 时 Y 的条件方差 $Var(Y|x)$ 是常数
- 二元正态分布（bivariate normal distribution）的正式定义
- 当 X 和 Y 具有零相关的二元正态分布，那么 X 和 Y 独立

## 给定 X 时 Y 的条件分布

首先假设：

1. 对每个 x，连续随机变量 Y 服从正态分布
2. 给定 x 时 Y 的条件均值 $E(Y|x)$ 在 x 上是线性的。即

$$E(Y|x)=\mu_Y+\rho \dfrac{\sigma_Y}{\sigma_X}(x-\mu_X)$$
3. 对每个x，Y 的条件方差 $Var(Y|x)=\sigma^2_{Y|X}$ 是常数

如下图所示：

![[Pasted image 20230529135748.png|350]]
蓝色线表示给定 x 时 Y 的条件均值与 x 的线性关系。对指定高度 x，如 $x_1$，红色点表示该 x 值对应的所有可能的体重 y 的取值。注意，每个 x 值的红点范围都是相同的，这是因为我们假设对每个 x 条件方差 $\sigma^2_{Y|X}$ 相同。如果把这个二维图变成三维图，则每组红点对应完全相同的正态分布曲线。

![[Pasted image 20230529141002.png|350]]
综上所述，给定 $X=x$ 时 Y 的条件分布为：

$$Y|x \textasciitilde N\left(\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X), \quad??\right)$$
如果能获得给定 X 时 Y 的条件方差 $\sigma^2_{Y|X}$，即能根据正态分布计算条件概率 $P(140<Y<160|X=x)$。

以下定理可以解决该问题：

如果给定 $X=x$ 时 Y 的条件分布 Y 服从均值为 $\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)$ 方差为 $\sigma^2_{Y|X}$ 的正态分布，则条件方差为：

$$\sigma^2_{Y|X}=\sigma^2_Y(1-\rho^2)$$
**证明：** 因为 Y 的连续随机变量，所以使用连续随机变量的条件方差公式：

$$\sigma^2_{Y|X}=Var(Y|x)=\int^{\infty}_{-\infty}(y-\mu_{Y|X})^2h(y|x)dy$$
替代 $\mu_{Y|X}=\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)$ 得到：

$$\sigma^2_{Y|X}=\int^{\infty}_{-\infty}[y-\mu_Y-\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)]^2h(y|x)dy$$
两边同时乘以 $f_X(x)$ 并对 x 积分，得到：

$$\int^{\infty}_{-\infty}\sigma^2_{Y|X}f_X(x)dx=\int^{\infty}_{-\infty}\int^{\infty}_{-\infty}[y-\mu_Y-\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)]^2h(y|x)f_X(x)dydx$$
因为 $\sigma^2_{Y|X}$ 是常量，不依赖于 x，因此可以将其从积分中提出来。而右侧 $h(y|x)f_X(x)$ 为 $f(x,y)$，因此：

$$\sigma^2_{Y|X}\int^{\infty}_{-\infty}f_X(x)dx=E\{[(Y-\mu_Y)-\left(\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)\right)]^2\}$$


