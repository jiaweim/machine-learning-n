# 卡方分布

- [卡方分布](#卡方分布)
  - [简介](#简介)
  - [参考](#参考)

2020-04-16, 18:52
***

## 简介

卡方分布是一种抽样分布。

设 $X_1,\cdots,X_n$ 相互独立，都是来自总体 $N(0,1)$ 的样本，则称统计量：

$$
\chi^2=X_1^2+X_2^2+\cdots+X_n^2
$$

服从自由度为 $n$ 的 $\chi^2(n )$ 分布。其概率密度函数为：

$$
f(x)=\begin{cases}
    \frac{1}{2^{n/2}\Gamma(n/2)}x^{n/2-1}e^{-x/2}\\
    0
\end{cases}
$$

分布图如下所示（k 为自由度）：

<img src="./images/600px-Chi-square_pdf.svg.png" alt="File:Chi-square pdf.svg" style="zoom:80%;" />

卡方分布具有如下重要性质：

1. 设 $X_1$, $X_2$ 独立，$X_1～\chi^2(m)$, $X_2～\chi^2(n)$，则 $X_1+X_2～\chi^2(m+n)$

直接从卡方分布的定义，很容易证明该式。

2. 若 $X_1,\cdots,X_n$ 独立，且都服从指数分布，则

$$
X=2\lambda (X_1+\cdots+X_n) ～\chi^2(2n)
$$



## 参考

- https://en.wikipedia.org/wiki/Chi-squared_distribution