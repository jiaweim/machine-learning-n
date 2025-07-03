# Levenberg-Marquardt

## 简介

在数学和计算领域，莱文博格-马夸尔特（Levenberg-Marquardt Algorithm, LMA）用于解决非线性最小二乘问题，又称为阻尼最小二乘法（Damped Least-Squares, DLS）。这类最小化问题在最小二乘曲线拟合中很常见，即将参数化数学模型拟合到一组数据点，通过最小化一个目标函数，该目标函数表示为模型函数与一组数据点之间的误差平方和。如果模型的系数是线性的，那么最小二乘目标函数是系数的二次函数。该目标函数可以通过线性方程组一步求解。如果拟合函数的系数不是线性的，则最小二乘问题需要迭代求解。此类算法通过对模型系数值进行一系列精心更新，来降低模型函数与数据点之间的误差平方和。

LMA 算法结合了两种数值最小化算法：高斯-牛顿算法（Gauss-Newton Algorithm, GNA）和梯度下降（Gradient-Descent）。在梯度下降中，通过更新最陡下降方向的系数来降低误差平方和；在高斯-牛顿法中，假设最小二乘函数的系数为局部二次函数，并求解二次函数的最小值以最小化误差平方和：

- 当系数远离其最优值时，LMA 的行为与梯度下降类似
- 当系数接近最优值时，LMA 的行为更类似高斯-牛顿法

LMA 比 GNA 更稳健，在许多情况下，即使起始值距离最终最小值很远，它也能找到解。对行为良好的函数和合理的起始参数，LMA 通常比 GNA 慢。

LMA 算法最初由 Kenneth Levenberg 于 1944 年在法兰克福陆军兵工厂工作时发表。1963 年，杜邦公司的统计学家 Donald Marquardt 重新发现该算法，之后， Girard, Wynne, Morrison 也分别独立发现了该算法。

## 问题


Levenberg-Marquardt 算法主要应用于最小二乘曲线拟合问题：给定一组 $m$ 对自变量和因变量对 $(x_i,y_i)$，求出模型曲线 $f(x,\beta)$ 的参数 $\beta$，使用残差平方和 $S(\beta)$ 最小化：

$$
\hat{\beta}\in \argmin_{\beta}S(\beta)\equiv \argmin_{\beta}\sum_{i=1}^m[y_i-f(x_i,\beta)]^2
$$

## 解

与其它数值最小化算法一样，LMA 是一个迭代过程。用户提供参数的初始值 $\beta$。当只有一个最小值，使用 $\beta^T=(1,1,\cdots,1)$ 这种猜测就可以正常工作；当存在多个最小值时，只有初始猜测值比较接近最优解时，算法才会收敛到全局最小值。

在每次迭代时，参数向量 $\beta$ 被新的估计值 $\beta+\sigma$ 替代。为了确定 $\sigma$ 值，使用线性化来近似 $f(x_i),\beta+\sigma$：

$$
f(x_i,\beta+\sigma)\approx f(x_i,\beta)+J_i\sigma
$$

其中：

$$
J_i=\frac{\partial f(x_i,\beta)}{\partial \beta}
$$

即 $f(x_i,\beta)$ 关于 $\beta$ 的梯度。

残差平方和 $S(\beta)$ 在相对 $\beta$ 的梯度为 0 时达到最小。带入一阶导数近似，得到：

$$
S(\beta+\sigma)\approx \sum_{i=1}^m [y_i-f(x_i,\beta)-J_i\delta]^2
$$


## 参考

- https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
- https://people.duke.edu/~hpgavin/lm.pdf
