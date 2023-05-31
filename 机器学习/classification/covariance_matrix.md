# 协方差矩阵

## 简介

我们可以用均值、中位数和方差总结包含单个特征的数据集；用协方差和相关系数总结包含两个特征的数据集。那么对包含更多特征的数据集呢？

对包含多个特征的数据集，将数据建模为 d 维随机向量。其均值定义为每个分量的均值组成的向量。

**定义 1.1** （随机变量的均值）d 维随机向量 $\tilde{x}$ 的均值为：

$$E(\tilde{x}) \coloneqq \begin{bmatrix}
E(\tilde{x}[1]) \\
E(\tilde{x}[2]) \\
\cdots \\
E(\tilde{x}[d])
\end{bmatrix}\tag{1}$$
类似地，我们可以将随机矩阵的均值定义为矩阵每个元素的均值组成的矩阵。

**定义 1.2** （随机矩阵的均值）$d_1\times d_2$ 随机矩阵的均值为：

$$E(\tilde{x}) \coloneqq \begin{bmatrix}
E(\tilde{X}[1,1]) & E(\tilde{X}[1,2]) & \cdots & E(\tilde{X}[1,d_2]) \\
E(\tilde{X}[2,1]) & E(\tilde{X}[2,2]) & \cdots & E(\tilde{X}[2, d_2]) \\
\vdots & \vdots & \vdots & \vdots \\
E(\tilde{X}[d_1, 1]) & E(\tilde{X}[d_1, 2]) & \cdots & E(\tilde{X}[d_1, d_2])
\end{bmatrix}\tag{2}$$

期望的线性特性也适用于随机向量和随机矩阵。

**引理 1.3** （随机向量和随机矩阵的期望的线性）设 $\tilde{x}$ 为 d 维随机向量，$b\in\mathbb{R}^m$, $A\in \mathbb{R}^{m\times d}$ ，其中 $m$ 为正整数，则：

$$E(A\tilde{x}+b)=AE(\tilde{x}+b) \tag{3}$$

类似地，设 $\tilde{X}$ 是 $d_1\times d_2$ 随机矩阵，$B\in \mathbb{R}^{m\times d_2}$，$A\in \mathbb{R}^{m\times d_1}$，其中 $m$ 为正整数，则：

$$E(A\tilde{X}+B)=AE(\tilde{X})+B \tag{4}$$
**证明:** 下面对向量的结果进行证明，矩阵的证明是一样的。$E(A\tilde{x}+b)$ 的第 $i$ 项等于：

$$
\begin{align}
E(A\tilde{x}+b)[i]&= E((A\tilde{x}+b)[i]) \qquad \text{根据随机向量的均值定义} \tag{5} \\
&=E\left(\sum^{d}_{j=1}A[i,j]\tilde{x}[j]+b[i]\right) \tag{6} \\
&=\sum^{d}_{j=1}A[i,j]E(\tilde{x}[j])+b[i] \qquad \text{根据标量期望的线性} \tag{7} \\
&=(AE(\tilde{x})+b)[i] \tag{8}
\end{align}
$$
我们通常通过计算随机向量的样本均值来估计总体均值。

**定义 1.4**（多变量（multivariate）数据的样本均值）设 $X \coloneqq \{x_1,x_2,...,x_n\}$ 为 d 维实数向量集合。则样本均值为：

$$\mu_X \coloneqq \frac{\sum^{n}_{i=1} x_i}{n} \tag{9}$$
在概率模型中使用随机向量，知道其线性组合的方差非常有用，即对确定向量 $v \in \mathbb{R}^d$，随机变量 $\langle v,\tilde{x}\rangle$ 的方差。根据期望的线性特性，该方差为：

$$
\begin{align}
Var(v^T\tilde{x})&=E\left((v^T\tilde{x}-E(v^T\tilde{x}))^2\right) \tag{10} \\
&= E\left((v^Tc(\tilde{x}))^2\right) \tag{11}\\
&=v^TE\left(c(\tilde{x})c(\tilde{x})^T\right)v \tag{12}
\end{align}
$$
其中 $c(\tilde{x}) \coloneqq \tilde{x}-E(\tilde{x})$ 为中心化随机向量。例如，当 $d=2$，且 $\tilde{x}$ 的均值为 0，则有：

$$
\begin{align}
E(c(\tilde{x})c(\tilde{x})^T)&=E(\tilde{x}\tilde{x}^T) \tag{13} \\
&=E\left(\begin{bmatrix}
\tilde{x}[1]\\
\tilde{x}[2]
\end{bmatrix}
\begin{bmatrix}
\tilde{x}[1] & \tilde{x}[2]
\end{bmatrix}\right) \tag{14} \\
&=E\left(\begin{bmatrix}
\tilde{x}[1]^2 & \tilde{x}[1]\tilde{x}[2]\\
\tilde{x}[1]\tilde{x}[2] & \tilde{x}[2]^2
\end{bmatrix}\right) \tag{15} \\ 
&=\begin{bmatrix}
E(\tilde{x}[1]^2) & E(\tilde{x}[1]\tilde{x}[2]) \\
E(\tilde{x}[1]\tilde{x}[2]) & E(\tilde{x}[2]^2)
\end{bmatrix}\tag{16} \\
&=\begin{bmatrix}
Var(\tilde{x}[1]) & Cov(\tilde{x}[1], \tilde{x}[2]) \\
Cov(\tilde{x}[1],\tilde{x}[2]) & Var(\tilde{x}[2])
\end{bmatrix} \tag{17}
\end{align}
$$
由此得到随机向量的协方差矩阵。

**定义 1.5** （协方差矩阵）d 维随机向量 $\tilde{x}$ 的协方差矩阵是一个 $d\times d$ 矩阵：

$$\begin{align}
\sum_{\tilde{x}} &\coloneqq E(c(\tilde{x})c(\tilde{x})^T) \tag{18} \\
&=\begin{bmatrix}
Var(\tilde{x}[1]) & Cov(\tilde{x}[1],\tilde{x}[2]) & \cdots & Cov(\tilde{x}[1],\tilde{x}[d]) \\
Cov(\tilde{x}[1],\tilde{x}[2]) & Var(\tilde{x}[2] & \cdots & Cov(\tilde{x}[2],\tilde{x}[d]) \\
\vdots & \vdots & \ddots & \vdots \\
Cov(\tilde{x}[1],\tilde{x}[d]) & Cov(\tilde{x}[2],\tilde{x}[d]) & \dots & Var(\tilde{x}[d])
\end{bmatrix} \tag{19}
\end{align}$$

其中 $c(\tilde{x}) \coloneqq \tilde{x} - E(\tilde{x})$。

协方差矩阵编码随机向量的任意线性组合的方差。

**引理 1.6** 对任意随机向量 $\tilde{x}$，其协方差矩阵 $\sum_{\tilde{x}}$，以及任意向量 $v$,

$$Var(v^T\tilde{x})=v^T\sum_{\tilde{x}}v \tag{20}$$
证明：公式 （12）直接导出。


## 如何计算协方差矩阵




## 参考

- https://cds.nyu.edu/wp-content/uploads/2021/05/covariance_matrix.pdf