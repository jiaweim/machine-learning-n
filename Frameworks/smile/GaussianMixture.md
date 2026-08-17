# 混合高斯模型

## 简介

混合高斯模型（Gaussian Mixture Model, GMM）是一种概率密度估计方法，它假设观测数据由 K 个参数未知的高斯分布按一定权重混合生成。核心目标是用这 $K$ 个高斯分量（均值 $\mu$，协方差 $\sum$，权重 $\pi$）拟合复杂数据分布，实现**聚类**或**密度估计**。

**模型假设**

数据分布：
$$
p(x)=\sum_{k=1}^K\pi_kN(x|\mu_k,Σ_k)
$$
其中，$\pi_k\ge 0$，$\sum_{k=1}^K \pi_k=1$。

**关键难点**

对数似然隐变量没有解析解，需要用 EM 算法迭代求解。

## smile-core

`smile.stat.distribution.GaussianMixture` 提供高斯混合模型的实现。