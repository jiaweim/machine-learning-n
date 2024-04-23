# 鲁棒回归（Robust Regression）

## 简介

我们已经了解普通最小二乘来估计回归线，然而，普通最小二乘无法处理非恒定方差和异常值问题，此时需要不同的方法来估计回归线。普通最小二乘描述：

$$
\begin{align*} \hat{\beta}_{\textrm{OLS}}&=\arg\min_{\beta}\sum_{i=1}^{n}\epsilon_{i}^{2} \\ &=(\textbf{X}^{\textrm{T}}\textbf{X})^{-1}\textbf{X}^{\textrm{T}}\textbf{Y} \end{align*}
$$

主要内容：

- 加权最小二乘的基本思想；
- 将加权最小二乘应用于方差不稳定的回归实例；
- 鲁棒回归的目的。

 ## 鲁棒回归方法

 

## 参考

- https://online.stat.psu.edu/stat501/lesson/topic-1-robust-regression