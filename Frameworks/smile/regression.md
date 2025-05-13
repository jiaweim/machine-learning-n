# 回归

## 简介

与分类问题不是，回归分析输出连续值。Smile 的回归算法位于 `smile.regression`  包中，所有算法都实现了 `Regression` 接口，该接口包含一个 `predict` 方法，用于将模型应用于实例。所有算法都可以通过构造函数进行模型训练；同事，每个算法都有一个对应的 `Trainer` 类，用于保存模型超参数并应用于多个训练数据集。

高级运算符定义咋 Scala trait `smile.regression.Operators` 中，也定义在 `smile.regression` 包中。下面讨论每种算法、它们的高级 Scala API 以及示例。

## 普通最小二乘法

在线性回归中，因变量为自变量的线性组合。残差是模型预测值与真实值之间的差值。普通最小二乘法获取使残差平方和（Sum of Squared Residues, SSE）最小的参数。

当自变量

## 参考

- https://haifengl.github.io/regression.html