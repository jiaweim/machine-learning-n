# EM 算法及其推广

***
## 简介

EM 算法是一种迭代算法，1977 年由 Dempster 等人总结提出，用于含有隐变量（hidden variable）的概率模型参数的极大似然估计，或极大后验概率估计。EM算法的每次迭代由两步组成：E步，求期望（expectation）；M步，求极大（maximization）。所以这一算法称为期望极大算法（expectation maximization algorithm），简称EM算法。

```ad-tip
隐变量：不能被直接观察到，但是对系统的状态和能观察到的输出存在影响的因素。
```

## EM 算法的引入

概率模型有时既含有观测变量（observable variable），又含有隐变量或潜在变量（latent variable）。如果概率模型的变量都是观测变量，那么给定数据，可以直接用极大似然估计法，或贝叶斯估计法估计模型参数。但是，当模型含有隐变量时，就不能简单地使用这些估计方法。EM算法就是含有隐变量的概率模型参数的极大似然估计法，或极大后验概率估计法。我们仅讨论极大似然估计，极大后验概率估计与其类似。

### EM 算法

首先介绍一个使用EM算法的例子。

**例 9.1（三硬币模型)**

假设有3枚硬币，分别记作 A, B, C。这些硬币正面出现的概率分别是 π, p 和 q。进行如下掷硬币试验：先掷硬币 A，根据其结果选出硬币 B 或硬币 C，正面选硬币 B，反面选硬币 C；然后掷选出的硬币，掷硬币的结果，出现正面记作 1，出现反面记作 0；独立地重复 n 次试验（这里，n=10)，观测结果如下：

```
1,1,0,1,0,0,1,0,1,1
```

假设只能观测到掷硬币的结果，不能观测掷硬币的过程。问如何估计三硬币正面出现的概率，即三硬币模型的参数。

**解**：三硬币模型可以写作

$$\begin{equation}
\begin{split}
P(y|\theta)&=\sum_z P(y,z|\theta)\\
&=\sum_z P(z|\theta)P(y|z,\theta)\\
&=\pi p^y(1-p)^{1-y}+(1-\pi)q^y(1-q)^{1-y}
\end{split}
\end{equation}\tag{1}$$
这里，随机变量 $y$ 是观测变量，表示一次试验观测的结果是 1 或 0；随机变量 $z$ 是隐变量，表示未观测到的掷硬币 A 的结果；$\theta=(\pi,p,q)$ 是模型参数。这一模型是以上数据的生成模型。注意，随机变量y的数据可以观测，随机变量之的数据不可观测。

将观测数据表示为 $Y=(Y_1, Y_2,...,Y_n)^T$，未观测数据表示为 $Z=(Z_1, Z_2,...,Z_n)^T$，则观测数据的似然函数为

$$P(Y|\theta)=\sum_Z P(Z|\theta)P(Y|Z,\theta) \tag{2}$$
即：

$$P(Y|\theta)=\prod^{n}_{j=1}[\pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_i}] \tag{3}$$


## EM算法在高斯混合模型学习中的应用

EM算法的一个重要应用是高斯混合模型的参数估计。高斯混合模型应用广泛，在许多情况下，EM算法是学习高斯混合模型(Gaussian mixture model)的有效方法。

### 高斯混合模型

**定义2（高斯混合模型）**

高斯混合模型是指具有如下形式的概率分布模型：

$$P(y|\theta)=\sum^{K}_{k=1}\alpha_k\phi(y|\theta_k) \tag{24}$$
其中，$\alpha_k$ 是系数，$\alpha_k\ge 0$，$\sum^{K}_{k=1}\alpha_k=1$；$\phi(y|\theta_k)$ 是高斯分布密度，$\theta_k=(\mu_k,\sigma^2_k)$，

$$\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}exp(-\frac{(y-\mu_k)^2}{2\sigma^2_k})\tag{25}$$
称为第 $k$ 个分模型。

一般混合模型可以由任意概率分布密度代替式（25）中的高斯分布密度，这里只介绍最常用的高斯混合模型。

### 高斯混合模型参数估计的 EM 算法

假设观测数据 $y_1,y_2,...,y_N$ 由高斯混合模型生成，

$$P(y|\theta)=\sum^{K}_{k=1}\alpha_k\phi(y|\theta_k) \tag{26}$$
其中，$\theta=(\alpha_1,\alpha_2,...,\alpha_K;\theta_1,\theta_2,...,\theta_K)$。我们用 EM 算法估计高斯混合模型的参数 $\theta$。

1. 明确隐变量，写出完全数据的对数似然函数

可以设想观测数据 $y_j$, $j=1,2,...,N$ 是这样产生的：首先依概率 $\alpha_k$ 选择第 $k$ 个高斯分布分模型 $\phi(y|\theta_k)$，然后依第 $k$ 个分模型的概率分布 $\phi(y|\theta_k)$ 生成观测数据 $y_i$。这时观测数据 $y_j,j=1,2,...,N$ 是已知的；反映观测数据 $y_j$ 来自第 $k$ 个分模型的数据是未知的，$k=1,2,…,K$，以隐变量 $\gamma_{jk}$ 表示，其定义如下：

$$\gamma_{jk}=\begin{cases}
1, &\text{第 j 个观测来自第 k 个分模型}\\
0, &\text{否则}
\end{cases}\tag{27}
$$
其中，$j=1,2,..,N; k=1,2,...,K$ ，$\gamma_{jk}$ 是 0-1 随机变量。

有了观测数据 $y_j$ 及未观测数据 $\gamma_{jk}$，那么完全数据是

$$(y_j,\gamma_{j1},\gamma_{j2},...,\gamma_{jK}), j=1,2...,N$$
于是，可以写出完全数据的似然函数：

$$\begin{equation}
\begin{split}
P(y|\gamma|\theta)
&=\prod^{N}_{j=1}P(y_j,\gamma_{j1},\gamma_{j2},...,\gamma_{jK}|\theta)\\
&=\prod^{K}_{k=1}\prod^{N}_{j=1}[\alpha_k\phi(y_j|\theta_k)]^{\gamma_{jk}}\\
&=\prod^{K}_{k=1}\alpha_k^{n_k}\prod^{N}_{j=1}[\phi(y_j|\theta_k)]^{\gamma_{jk}}\\
&=\prod^{K}_{k=1}\alpha_k^{n_k}\prod^{N}_{j=1}[\frac{1}{\sqrt{2\pi}\sigma_k}exp(-\frac{(y_j)-\mu_k)^2}{2\sigma_k^2})]^{\gamma_{jk}}
\end{split}
\end{equation}$$


## 参考

- 《统计学习方法, 2ed》，李航，清华大学出版社