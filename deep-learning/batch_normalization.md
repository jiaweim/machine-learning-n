# Batch Normalization

## 简介

Batch Normalization（简称 Batch Norm）是 2015 年提出的的方法。看一下机器学习比赛的结果，就会发现很多通过使用这个方法而获得优异结果的例子。

Batch Norm 有以下优点：

- 可以使学习快速进行（可以增大学习率）；
- 不那么依赖初始值；
- 抑制过拟合（降低 Droupout 等的必要性）。

Batch Norm 的思路是调整各层的激活值分布使其拥有适当的广度。为此，要向神经网络中插入对数据分布进行正规化的层，即 Batch Norm 层。如下图所示：

![](images/2021-08-13-15-01-53.png)

Batch Norm，顾名思义，以进行学习时的 mini-batch 为单位，按 mini-batch 进行正则化。具体而言，就是使数据分布的均值为 0、方差为 1。用数学式表示如下：

$$
\mu_B \leftarrow \frac{1}{m}\sum_{i=1}^m x_i \tag{1}
$$

$$
\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2 \tag{2}
$$

$$
\hat{x}_i \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}} \tag{3}
$$

式（1）计算 mini-batch 的均值，式（2）计算 mini-batch 的方差，式（3）对输入进行均值为 0、方差为 1 的正则化。

式（3）中 $\epsilon$ 是一个微小值，用于防止出现除以 0 的情况。

接着，Batch Norm 层会对正则化后的数据进行缩放和平移变换，用数学式可以如下表示：

$$
y_i \leftarrow \gamma \hat{x}_i+\beta \tag{4}
$$

这里，γ 和 β 是参数，一开始，γ=0，β=1，然后再通过学习调整到合适的值。
