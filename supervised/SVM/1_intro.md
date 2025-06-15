# 支持向量机

2021-02-20 ⭐
@author Jiawei Mao

***

## 概述

支持向量机（Support vector machine, SVM）是一种监督学习算法，是一种稳健的分类和回归技术，在不过度拟合训练数据的情况下，最大化模型的预测精度。

SVM 是由 Vladimir Vapnik 及其同事 1955 年在贝尔实验室基于结构风险最小化原理开发的统计学习理论。

将输入空间非线性的映射到高维特征空间（kernel）。

- 对分类问题，SVM在空间中构造一个最佳的分离超平面；
- 对回归问题，在该空间执行线性回归。

SVM 在很多领域都有应用，包括客户关系管理（CRM）、人脸和其它图像识别、生物信息学、文本挖掘概念提取、入侵检测、蛋白质结构预测和语音识别等。

**SVM 缺点**：SVM 很难扩展到大型数据集，并且在图像分类等感知问题上的效果不好。SVM 是一种比较浅层的方法，因此要将其应用于感知问题，首先要手动提取有用的表示（特征工程），这一步很难，而且不稳定。

## SVM 算法

支持向量机（Support Vector Machine, SVM）是由分离超平面（hyperplane）定义的判别式分类器。即给定带标签的训练数据（监督学习），该算法会输出对样本进行分离的最佳超平面。在二维空间中，超平面是将平面分为两部分的线。


如下如所示，假设你有两类带有标签的数据：

<img src="images/2019-10-30-14-30-15.png" style="zoom:50%;" />

在上面绘制一条线，最好的分开两部分，对二维空间，这条线就是我们需要的超平面，如下所示：

<img src="images/2019-10-30-14-31-21.png" style="zoom:50%;" />

所以 SVM 就是从标记数据中找到最佳分离不同类别的超平面。

## 核函数

我们很可能碰到使用直线无法对数据进行分类的情况，例如：

<img src="images/2019-10-30-14-33-46.png" style="zoom:50%;" />

此时该怎么办？我们可以再添加一条 z 轴（径向坐标系），并假设 z 轴上点的坐标为 $w=x^2+y^2$，此时根据数据点到 z 轴的距离可以很好地对数据进行分离，如下所示：

<img src="images/2019-10-30-14-37-33.png" style="zoom:50%;" />

这种进行转换的函数（$w=x^2+y^2$）就是**核**（kernels）。

然后我们将该线转换回原来的平面：

<img src="images/2019-10-30-14-45-48.png" style="zoom:50%;" />

## 数据重叠

如果数据之间重叠了怎么办？例如：

<img src="images/2019-10-30-14-47-33.png" style="zoom:50%;" />

你可能会这么绘制：

<img src="images/2019-10-30-14-48-58.png" style="zoom:50%;" />

也可能这么绘制：

<img src="images/2019-10-30-14-48-10.png" style="zoom:50%;" />

两种方式都行，此时就需要取舍了。

## 调整参数

### 内核（Kernel）
在线性 SVM 中，通过线性代数对数据进行转换学习超平面。用于转换的函数称为**内核函数**（kernel function）。常用的内核有如下几种：
- Linear
- Polynominal
- Radial basis function (RBF)
- Sigmoid

多项式内核和指数内核在更高维度计算分离线。

### 线性内核

线性内核（linear kernel）使用输入 (x) 和支持向量（$x_i$）的点乘执行预测：

$$
f(x)=B_0+sum(a_i*(x,x_i))
$$

该公式计算输入向量 $x$ 和所有的支持向量 $x_i$ 的点乘。参数 $B_0$ 和 $a_i$ 通过训练数据学习获得。

### 多项式内核

多项式内核（polynominal kernel）可以写为：

$$
K(x,x_i)=1+sum(x*x_i)^d
$$

### 指数内核

$$
K(x,x_i)=exp(-gamma*sum(x-x_i^2))
$$

## 正则化

正则化参数（Regularization parameter, C），在 sklearn 中一般称为 C 参数，是告诉 SVM 对每个训练样本你希望避免错误分类的程度。
- C 值较大，表示尽可能好的分离数据，如果超平面能够很好的分离训练数据，优化器会选择一个边距较小的超平面；
- 相反，如果 C 值很小，优化器会寻找分隔较大的超平面，即使该超平面会导致更多错误的分类。

如下所示：

|C 值小|C 值大|
|---|---|
|<img src="images/2019-10-30-15-07-27.png" style="zoom:50%;" />|<img src="images/2019-10-30-15-07-47.png" style="zoom:50%;" />|

两个可以认为是不同正则化参数导致的结果。左侧的 C 值较小，追求较大的超平面，有部分错误分类；右侧的 C 值较大，分离好。

在保证分类准确性的前提下尽可能取小的C值，有利于模型的稳健性。

## Gamma

Gamma 值定义单个训练样本影响：
- gamma 值低表示远离分离线的数据点在计算分离线时也考虑。
- gamma 值高表示在计算时只考虑挨着分离线的点。

如下所示：

<img src="images/2019-10-30-15-14-44.png" style="zoom:50%;" />

<img src="images/2019-10-30-15-15-03.png" style="zoom:50%;" />

## Margin

SVM 的核心是获得一个较好的分离面。

边距（Margin）是分离线和最接近点的距离。位于边距上的数据点称为**支持向量**。

边距越宽，模型在预测新数据的类别时就越好。

一个好的边距和两个分类的距离都足够大。下图显示好的和不好边距的示例：

<img src="images/2019-10-30-15-17-58.png" style="zoom:50%;" />

<img src="images/2019-10-30-15-18-16.png" style="zoom:50%;" />

下面这个就有点过拟合了，容许少量的错误分类以获得更宽的边距往往有益于新数据的预测。

## One-Class SVM

One Class SVM 指 training data 只有一类 positive (或 negative) 的数据，而没有另外一类。此时，你需要 learn 的是 training data 的 boundary，所以不能使用 maximum margin，因此你没有两类 data。如下图所示：

<img src="images/2019-12-31-09-25-32.png" style="zoom:50%;" />

Scholkopf 假设最好的 boundary 要远离 feature space 中的原点。左边是 original space 中的 boundary，无数同心圆可以有无数的 boundary，但是比较靠谱的是找一个比较紧的boundary（红色）。这个目标转到到 feature space 就是找一个离原点比较远的 boundary，同样是红色的直线。

## 参考

- [Introduction to Support Vector Machine](https://ai.plainenglish.io/introduction-to-support-vector-machine-svm-cd0759098471)