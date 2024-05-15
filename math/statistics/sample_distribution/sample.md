# 样本

## 概念

试验的全部可能得观察值称为**总体**。每一个可能观察值称为**个体**。总体所包含的个体数量称为总体的容量：

- 容量有限的称为**有限总体**
- 容量无限的称为**无限总体**

总体中每一个个体是随机试验的一个观察值，对应一个随机变量 $X$ 的值，因此，一个总体对应一个随机变量 $X$。对总体的研究就是对随机变量 $X$ 的研究。

在实际中，总体的分布一般是未知的，或只知道它的形式而不知道参数值。在数理统计中，人们通过从总体中抽取一部分个体，根据获得数据来对总体分布做出判断，倍抽出的部分个体叫作总体的一个**样本**。

## 抽样分布

样本是进行统计推断的依据。

**定义**

设 $X_1,X_2,\cdots,X_n$ 是来自总体 $X$ 的一个样本，$g(X_1,X_2,\cdots,X_n)$ 是 $X_1,X_2,\cdots,X_n$ 的函数，若 $g$ 中不包含未知参数，就称 $g(X_1,X_2,\cdots,X_n)$ 是 $X_1,X_2,\cdots,X_n$ 是一个**统计量**。

因为 $X_1,X_2,\cdots,X_n$ 都是随机变量，而统计量 $g(X_1,X_2,\cdots,X_n)$ 是 $X_1,X_2,\cdots,X_n$ 是随机变量的函数，因此统计量是一个随机变量。

设 $x_1, x_2,\cdots,x_n$ 是样本 $X_1,X_2,\cdots,X_n$ 的样本值，则称 $g(x_1,x_2,\cdots,x_n)$ 是 $g(X_1,X_2,\cdots,X_n)$ 是 $X_1,X_2,\cdots,X_n$ 的观察值。

设 $X_1,X_2,\cdots,X_n$ 是来自总体 $X$ 的一个样本，$x_1, x_2,\cdots,x_n$ 是这个样本的观察值。定义

**样本平均值**

$$
\overline{X}=\frac{1}{n}\sum_{i=1}^n X_i
$$

**样本方差**

$$
S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2=\frac{1}{n-1}(\sum_{i=1}^n X_i^2-n\overline{X}^2)
$$

**样本标准差**

$$
S=\sqrt{S^2}=\sqrt{\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})^2}
$$

**样本 k 阶（原点）矩**

$$
A_k=\frac{1}{n}\sum_{i=1}^n X_i^k, \quad k=1,2,...
$$

**样本 k 阶中心矩**

$$
B_k=\frac{1}{n}\sum_{i=1}^n(X_i-\overline{X})^k,\quad k=2,3,...
$$

若总体 $X$ 的 k 阶矩 $E$