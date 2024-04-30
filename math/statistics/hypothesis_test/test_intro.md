# 假设检验

- [假设检验](#假设检验)
  - [1. 基本概念](#1-基本概念)
    - [零假设和备择假设](#零假设和备择假设)
    - [错误类型](#错误类型)
    - [统计检验和 p-value](#统计检验和-p-value)
    - [作出决定](#作出决定)
  - [2. 使用 p-Value 进行假设检验的步骤](#2-使用-p-value-进行假设检验的步骤)
    - [假设检验策略](#假设检验策略)

2024-04-30 ⭐
@author Jiawei Mao
***

## 1. 基本概念

### 零假设和备择假设

1. A **null hypothesis H0** is a statistical hypothesis that contains a statement of equality, such as $\le$, =, or $\ge$.
2. The **alternative hypothesis Ha** is the complement of the null hypothesis. 
It is a statement that must be true if H0 is false and it contains a statement of strict inequality, such as $>$, $≠$, or $<$.

例如：

$$
\begin{cases}
    H_0: \mu\le k\\
    H_a: \mu > k
\end{cases}
$$

$$
\begin{cases}
    H_0: \mu \ge k\\
    H_a: \mu < k
\end{cases}
$$

$$
\begin{cases}
    H_0: \mu = k\\
    H_a: \mu \ne k
\end{cases}
$$

### 错误类型

Because your decision is based on a sample rather than the entire population, there is always the possibility you will make the wrong decision.

**DEFINITION**

- A **type I error** occurs if the null hypothesis is rejected when it is true.
- A **type II error** occurs if the null hypothesis is not rejected when it is false.

如下表所示：
<img src="./images/image-20240430091340103.png" alt="image-20240430091340103" style="zoom:67%;" />

I 类错误：假阳；
II 类错误：假阴。

In a hypothesis test, the **level of significance** is your maximum allowable probability of making a type I error. It is denoted by a, the lowercase Greek letter alpha.

在假设检验中，**显著性水平**指发生第 I 类错误的最大概率，用字母 $\alpha$ 表示。

第 II 类错误的概率用字母 $\beta$ 表示。

将显著性水平设置为一个很小的值，表示希望假阳性的概率很小。三种常用的显著性水平：$\alpha=0.10$, $\alpha=0.05$, $\alpha=0.01$。

> 降低 $\alpha$，很可能增加 $\beta$。

### 统计检验和 p-value

声明零假设和备择假设，指定显著性水平，假设检验的下一步是随机抽样，获取样本统计量，如 $\overline{x}$, $\hat{p}$ 或 $s^2$。这些样本统计量称为**检验统计量（test statistic）**。

假设零假设为真，然后将检验统计量转换为标准化检验统计量，如 $z$, $t$ 或 $\chi^2$。

常见单样本统计检验：

|总体参数|Test statistic|Standardized test statistic|
|---|---|---|
|$\mu$|$\overline{x}$|$z$ ($\sigma$ 已知)<br />$t$ ($\sigma$ 未知)|
|$p$|$\hat{p}$|$z$|
|$\sigma^2$|$s^2$|$\chi^2$|

决定是否拒绝零假设的一种方法是：确定获得标准化检验统计量的概率是否小于显著性水平。

有三种类型的假设检验：

- left-tailed
- right-tailed
- two-tailed

1. 如果备择假设 $H_a$ 包含 $<$，则假设检验是**左边检验**

<img src="./images/image-20240430104419436.png" alt="image-20240430104419436" style="zoom:50%;" />

2. 如果备择假设 $H_a$ 包含 $>$，则假设检验是**右边假设**

<img src="./images/image-20240430105834537.png" alt="image-20240430105834537" style="zoom:50%;" />

3. 如果备择假设 $H_a$ 包含 $\ne$，则假设检验是**双边检验**

对双边检验，两边面积各为 $\frac{1}{2}p$

<img src="./images/image-20240430110300582.png" alt="image-20240430110300582" style="zoom:50%;" />

p-Value 越小，则拒绝 $H_0$ 的证据越充分。需要注意，即使 p-Value 非常非常小也不能证明 $H_0$ 是错误的，只能证明它可能是错误。

### 作出决定

假设检验的最后一步，作出决定，并解释为什么。对任何假设检验，只有两种结果：（1）拒绝原假设；（2）未能拒绝原假设。

**基于 p-Value 的决策**

通过比较 p-Value 和 $\alpha$ 进行判断：

1. 如果 $P\le \alpha$，则拒绝 $H_0$
2. 如果 $P>\alpha$，则未能拒绝 $H_0$

没有拒绝原假设并不代表接受原假设为 true。它仅仅表示没有足够的证据来拒绝原假设。

<img src="./images/image-20240430112405387.png" alt="image-20240430112405387" style="zoom:50%;" />

## 2. 使用 p-Value 进行假设检验的步骤

> **NOTE**
> 在进行假设检验时，应该在收集数据之前声明零假设和备择假设。

**假设检验步骤**

1. 用数学或语言方式声明假设。确定零假设和备择假设。

$$
H_0: ?
$$

$$
H_a: ?
$$

2. 指定显著性水平

$$
\alpha=?
$$

3. 确定标准化抽样分布，并绘图

<img src="./images/image-20240430113423974.png" alt="image-20240430113423974" style="zoom:67%;" />

4. 计算检验统计量和相应的标准化检验统计量。把它添加到上图中

<img src="./images/image-20240430113513050.png" alt="image-20240430113513050" style="zoom:67%;" />

5. 计算 p-Value
6. 作出决策

- 如果 p-Value 小于等于显著性水平，拒绝 $H_0$；
- 否则，未能拒绝 $H_0$

7. 作出声明，根据原假设解释决策结果。

### 假设检验策略

假设检验使用的策略取决于你是想支持还是拒绝一个声明。

假设检验是试图证明能够拒绝 $H_0$，所以：

- 如果你想**支持**某个说法，应该将其声明为**备择假设**
- 如果你想**拒绝**某个说法，应该将其声明为**零假设**