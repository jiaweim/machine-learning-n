# 假设检验

- [假设检验](#假设检验)
  - [1. 基本概念](#1-基本概念)
    - [相关概念](#相关概念)
  - [3. 功效函数](#3-功效函数)

2024-04-30 ⭐
@author Jiawei Mao
***

## 1. 基本概念

### 相关概念

在检验一个假设时所使用的统计量称为**检验统计量**。

使原假设得到接受的那些样本（$X_1,\cdots,X_n$）所在的区域 $A$，称为该检验的**接受域**；而使原假设被否定的那些样本所在的区域 $R$，则称为该检验的**拒绝域**。拒绝域又称为否定域，临界域。$A$ 与 $R$ 互补，知其一即知其二。定一个检验，等价于指定其接受域或拒绝域。

不论是原假设还是备择假设，若其中只包含一个参数值，就称为**简单假设**，否则就称为**复合假设**。

## 3. 功效函数

**定义：** 设 $\Phi$ 是原假设 $H_0$ 的一个检验，$\beta_{\Phi}(\theta_1,\cdots,\theta_k)$ 为其功效函数，如果：

$$
\beta_{\Phi}(\theta_1,\cdots,\theta_k)\le \alpha
$$

则称 $\Phi$ 为 $H_0$ 的执行水平为 $\alpha$ 的检验。


同一个原假设可以有许多检验方法，**功效函数** 用于区分不同检验方法的优劣。

**例如：** 假设元件寿命服从[指数分布](../distribution/exponential.md)，通过抽样数据判断“元件平均寿命不小于 5000 小时”。取如下检验：

$$
\begin{cases}
    H_0:\mu \ge 5000\\
    H_a:\mu < 5000
\end{cases} \tag{1}
$$

原假设 $H_0$ 被接受与否取决于样本 $X_1,\cdots,X_n$，而样本是随机的。因此原假设被否定的概率为：

$$
\beta_{H_a}(\lambda)=P_{\lambda}(\overline{X}<C)
$$

其中 $\lambda$ 为总体参数。因为 $2\lambda(X_1+\cdots+X_n)～\chi^2(2n)$，分布函数记为 $K_{2n}$，则有：

$$
\begin{aligned}
\beta_{H_a}(\lambda)
&=P_{\lambda}(X_1+\cdots+X_n<nC)\\
&=P_{\lambda}(2\lambda(X_1+\cdots+X_n)<2\lambda nC)\\
&=K_{2n}(2\lambda nC)
\end{aligned} \tag{2}
$$

该值与 $\lambda$ 相关，随 $\lambda$ 上升而增加。因为 $\lambda$ 越大，离开原假设 $1/\lambda \ge 5000$ 越远，一个合理的检验方法就需要用更大的概率去否定它。

函数（2）就是检验（1）的功效函数。由此，提供一般定义：

**定义** 设总体分布包含若干个未知参数 $\theta_1,\cdots,\theta_k$。$H_0$ 是关于这些参数的一个原假设，设有样本 $X_1,\cdots,X_n$，而 $\Phi$ 是基于这些样本对 $H_0$ 所做的一个检验。则称检验 $\Phi$ 的功效函数为：

$$
\beta_{\Phi}(\theta_1,\cdots,\theta_k)=P_{\theta_1,\cdots,\theta_k}
$$

即 $H_0$ 被否定的概率。

当某一特定参数值 $(\theta_1^0,\cdots,\theta_k^0)$ 使 $H_0$ 成立，我们希望 $\beta_{\Phi}(\theta_1,\cdots,\theta_k)$ 尽可能小；反之，则希望 $\beta_{\Phi}(\theta_1,\cdots,\theta_k)$ 尽可能大。

同一个原假设的两个检验 $\Phi_1$ 和 $\Phi_2$，哪一个更符合这个要求，哪一个就更好。
