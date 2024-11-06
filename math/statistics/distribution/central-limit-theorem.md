# 中心极限定理

- [中心极限定理](#中心极限定理)
  - [简介](#简介)
  - [示例](#示例)
  - [参考](#参考)

***

## 简介

在客观实际中有许多随机变量，它们是由大量的相互独立的随机因素的综合影响所形成。而其中每一个因素在总的影响中所起的作用都是微小的。这种随机变量往往近似地服从正态分布。这种现象就是中心极限定理的客观背景，下面介绍三个常用的中心极限定理。

**中心极限定理**：所研究的随机变量如果是由大量独立的随机变量相加而成，那么它的分布近似于正态分布。



**定理一：独立同分布的中心极限定理**

设随机变量 $X_1, X_2,...,X_n$ 相互独立，服从同一分布，且具有数学期望和方差：$E(X_k)=\mu$，$D(X_k)=\sigma^2>0$, $(k=1,2,...)$，则随机变量之和 $\sum^{n}_{k=1}X_k$ 的标准化变量：

$$
\begin{equation}
\begin{split}
Y_n&=\frac{\sum^{n}_{k=1}X_k-E(\sum^{n}_{k=1}X_k)}{\sqrt{D(\sum^{n}_{k=1}X_k)}}
&=\frac{\sum^{n}_{k=1}X_k-n\mu}{\sqrt{n}\sigma}
\end{split}
\end{equation}
$$

的分布函数 $F_n(x)$ 对于任意 x 满足：

$$
\begin{aligned}
\lim_{n\rightarrow\infty}F_n(x)&=\lim_{n\rightarrow\infty}P\lbrace \frac{\sum^{n}_{k=1}X_k-n\mu}{\sqrt{n}\sigma}\le x\rbrace\\
&=\int^x_{-\infty}\frac{1}{\sqrt{2\pi}e^{-t^2/2}}dt\\
&=\Upphi(x)
\end{aligned} \tag{1}
$$
证明略。

也就是说，均值为 $\mu$，方差为 $\sigma^2>0$ 的独立同分布的随机变量 $X_1,X_2,...,X_n$ 之和 $\sum^{n}_{k=1}X_k$ 的标准化变量，当 n 充分大时，有

$$
\frac{\sum^{n}_{k=1}X_k-n\mu}{\sqrt{n}\sigma}\approx N(0,1) \tag{2}
$$

在一般情况下，很难求出 n 个随机变量之和 $\sum^{n}_{k=1}X_k$ 的分布函数，式（2）表明，当 n 充分大时，可以通过 $\upphi(x)$ 给出其近似分布。这样，就可以利用正态分布对 $\sum^{n}_{k=1}X_k$ 作理论分析或实际计算，好处明显。

将（2）式左边改写成 $\frac{\frac{1}{n}\sum^{n}_{k=1}X_k-\mu}{\sigma/\sqrt{n}}=\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}$ ，这样，上述结果可写成：当 n 充分大时，

$$\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}\approx N(0,1) \tag{3}$$

这是独立同分布中心极限定理结果的另一个形式。即，均值为 $\mu$，方差为 $\sigma^2>0$ 的独立同分布的随机变量 $X_1,X_2,...,X_n$ 的算术平均 $\overline{X}=\frac{1}{n}\sum^{n}_{k=1}X_k$，当 n 充分大时近似地服从均值为 $\mu$，方差为 $\sigma^2/n$ 的正态分布。这一结果是数理统计中大样本统计推断的基础。

**定理二：李雅普诺夫（Lyapunov）定理**

设随机变量 $X_1,X_2,...,X_n$ 相互独立，它们具有数学期望和方差

$$E(X_k)=\mu_k$$
$$D(X_k)=\sigma^2_k>0$$
其中，$k=1,2,...$，记

$$B^2_n=\sum^{n}_{k=1}\sigma^2_k$$
若存在正数 $\delta$，使得当 $n\rightarrow \infty$ 时，

$$\frac{1}{B^{2+\delta}_n}\sum^{n}_{k=1}E\{|X_k-\mu_k|^{2+\delta}\}\rightarrow0$$

## 示例

**例 1**：一加法器同时收到 20 个噪声电压 $V_k (k=1,2,...,20)$，设它们是相互独立的随机变量，且都在区间 (0,10) 上服从均匀分布。记 $V=\sum^{20}_{k=1}V_k$，求 $P\{V>105\}$ 的近似值。

**解：** 易知 $E(V_k)=5$，$D(V_k)=100/12$，$k=1,2,...,20$。

## 参考

- https://online.stat.psu.edu/stat414/lesson/27
