# Markov Chain

## 简介

马尔科夫链（Markov Chain）指具有马尔科夫性质的随机变量序列。下面介绍一些统计中常用的相关概念。

## 定义

设 $\{X_n\}$ 是随机向量序列。当前仅当序列 $X_n$  中任意一项独立于 $X_{n-1}$ 前面的所有其它项，就称序列 $\{X_n\}$ 为马尔科夫链：
$$
F(x_n|x_{n-1},x_{n-2},\cdots,x_1)=F(x_n|X_{n-1})
$$
其中 $F$ 表示条件分布函数。

## 状态空间

马尔科夫链 $\{X_n\}$ 的状态空间 $S$ 是链中所有可能实现的集合。

下面按照难度递增的顺序介绍：

- 有限状态空间
- 无限但可数的状态空间
- 不可数状态空间

## 有限状态空间的马尔科夫链

假设状态空间为：
$$
S=\{s_1,\cdots,s_J\}
$$
即马尔科夫链中的取值有 $J$ 种可能：$s_1,\cdots,s_J$。

### Time-homogeneity

为链的第一个值指定**初始分布**，即长度为 $1\times J$ 包含初始概率 $\pi_1$ 的向量：
$$
P(X_1=s_1)=\pi_{11}\\
\cdots\\
P(X_1=s_J)=\pi_{1J}\\
$$
然后，选择 $J\times J$ 的**转移概率矩阵**（transition probability matrix, P）

## 参考

- https://www.statlect.com/fundamentals-of-statistics/Markov-chains