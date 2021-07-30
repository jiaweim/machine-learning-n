# 随机梯度下降

- [随机梯度下降](#随机梯度下降)
  - [简介](#简介)
  - [分类（Classification）](#分类classification)
  - [参考](#参考)

2021-07-26, 10:34
***

## 简介

随机梯度下降（Stochastic Gradient Descent, SGD）用于凸损失函数（convex loss function）下拟合线性分类器和回归的有效的方法，如支持向量机和逻辑回归。虽然 SGD 很早就应用在机器学习领域，但是最近在大规模学习中的应用受到广泛关注。

SGD 已成功应用于文本分类和自然语言处理中经常遇到的大规模和稀疏机器学习问题。由于数据稀疏，该模块中的分类器可以轻松扩展到超过 10^5 个训练样本和超过 10^5 个特征（features）的问题。

严格来说，SGD只是一种优化技术，并不对应于特定的机器学习模型。它只是一种训练模型的方法。在 scikit-learn 中，`SGDClassifier` 或 `SGDRegressor` 实例一般具有等价的估计器（estimator），但是可能使用不同的优化技术。例如，使用 `SGDClassifier(loss='log')` 得到逻辑回归，等价于使用 SGD 拟合 `LogisticRegression`。

SGD 的优点有：

- 高效
- 易于实现

SGD 的缺点有：

- SGD 需要许多超参数（hyperparameters），如正则化参数、许多次迭代；
- SGD 对 feature scaling 敏感

> 在拟合模型之前，要打乱数据，或者每次迭代之前设置 `shuffle=True`（默认使用）。理想情况下，还应该使用 `make_pipeline(StandardScaler(), SGDClassifier())` 归一化 features。

## 分类（Classification）



## 参考

- https://scikit-learn.org/stable/modules/sgd.html
