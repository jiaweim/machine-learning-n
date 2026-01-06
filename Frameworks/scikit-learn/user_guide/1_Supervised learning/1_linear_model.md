# 线性模型

@author Jiawei Mao
***

## 简介

下面介绍一系列用于线性回归分析的方法，其 target 值是一系列 feature 的线性组合，即广义的线性模型。在数学中，如果 $\hat{y}$ 是预测值，线性组合可以表示为：

$$
\hat{y}(w,x)=w_0+w_1x_1+...+w_px_p
$$

其中，向量 $w=(w_1,...,w_p)$ 为参数向量 `ceof_`，$w_0$ 为截距 `intercept_`。

使用广义线性模型进行分类，可以参考 [逻辑回归](#逻辑回归)。

## 1. 普通最小二乘法

⭐2026-01-06
`LinearRegression` 使用参数 $w=(w_1,...,w_p)$ 拟合线性模型，使得数据集 targets 的观测值和模型的预测值之间的残差平方和最小。从数学角度来说是解决如下问题：
$$
\min_{w} || X w - y||_2^2
$$

`LinearRegression` 的 `fit` 方法接收参数 $X$, $y$ 和 `sample_weight` 进行拟合，并将线性模型系数 $w$ 保存在 `coef_` 和 `intercept_` 属性中：

```python
>>> from sklearn import linear_model
>>> reg = linear_model.LinearRegression()
>>> reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression()
>>> reg.coef_
array([0.5, 0.5])
```

普通最小二乘（Ordinary Least Squares, OLS）的系数估计依赖于 features 的独立性。

当 feature 之间相关且设计矩阵（design matrix）$X$ 的部分 column 之间具有近似线性依赖关系，此时设计矩阵近似为奇异（singular）矩阵，导致最小二乘对 target 的随机误差十分敏感，产生较大的方差。在没有实验设计的情况下收集数据，可能会出现这种多重共线性（*multicollinearity*）的情况。

示例：

- [普通最小二乘和脊回归](../../examples/linear_model/plot_ols.md)

## 11. 逻辑回归

逻辑回归实现为 [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。虽然叫逻辑回归，但是它是线性分类模型。logistic-regression 在文献中也称为 logit-regression，maximum-entropy classification 或 log-linear classifier。在该模型中，使用逻辑函数建模单个试验的概率。

## 16. 稳健回归

稳健回归旨在拟合存在离群值（outlier）或模型错误的数据。

### 不同场景和概念

当处理包含离群值的数据，需要考虑以下几点：

- **离群值是 X 还是 y?**

|y 方向的离群值|X 方向的离群值|
|---|---|
|![y_outliers](./images/sphx_glr_plot_robust_fit_003.png)|![X_outliers](./images/sphx_glr_plot_robust_fit_002.png)|

- 离群值与误差幅度

离群点的数量很重要，离群点的离群幅度也很重要。

|Small outliers|Large outliers|
|---|---|
|![y_outliers](https://scikit-learn.org/stable/_images/sphx_glr_plot_robust_fit_003.png)|![large_y_outliers](https://scikit-learn.org/stable/_images/sphx_glr_plot_robust_fit_005.png)|

稳健拟合的一个重要概念是 breakdown-point：估计器在给出错误模型估计之前，离群数据的最大比例。breakdown-point 代表估计器对离群数据的最大容忍度。

> **注意**
> 在高维设置（`n_features` 很大）中进行稳健拟合非常困难，此时采用稳健模型可能效果不好。

**估计器选择**

Scikit-learn 提供了 3 个稳健回归估计器：RANSAC, Theil Sen 和 HuberRegressor

- `HuberRegressor` 通常比 `RANSAC` 和 `Theil Sen` 快，除非样本量很大，即 `n_samples` >> `n_features`。而且，默认参数下 `RANSAC` 和 `Theil Sen` 没有 `HuberRegressor` 稳健；
- `RANSAC` 比 `Theil Sen` 快，并且对样本数的变化 scale 更好；
- `RANSAC` 能更好处理 y 方向的较大离群值；
- `Theil Sen` 能更好处理 X 方向的中等大小离群值，但该优势在高维环境就没了。

如果不确定，就用 `RANSAC`。

### RANSAC

RANSAC（RANdom SAmple Consesnsus）随机样本一致性，

## 参考

- https://scikit-learn.org/stable/modules/linear_model.html
- https://www.mathsisfun.com/data/least-squares-regression.html
