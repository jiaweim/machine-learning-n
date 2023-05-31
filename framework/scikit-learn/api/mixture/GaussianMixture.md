# sklearn.mixture.GaussianMixture

2023-05-31
****
## 简介

```python
class sklearn.mixture.GaussianMixture(n_components=1, *, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
```

表示高斯混合模型的概率分布。该类用于估计高斯混合分布的参数。

## 参数

- **n_components**: int, default=1

cluster 数。

- **covariance_type**: {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’

协方差参数类型。

| 类型        | 说明                                  |
| ----------- | ------------------------------------- |
| "full"      | 每个 cluster 都有各自的协方差矩阵     |
| "tied"      | 所有 cluster 共享相同的协方差矩阵     |
| "diag"      | 每个 cluster 都有各自的对角协方差矩阵 |
| "spherical" | 每个 cluster 都有各自不同的简单协方差矩阵 |

球面协方差矩阵的非对角为 0，对角完全相同，即只有一个方差值。

- **tol**: float, default=1e-3

收敛阈值。当下界平面增益低于该阈值，EM 停止迭代。

- **reg_covar**: float, default=1e-6

添加到协方差对角线的非负正则化项。用于保证协方差矩阵元素值都是正数。

- **max_iter**: int, default=100

EM 迭代次数。

- **n_init**: int, default=1

执行的初始化次数，保留最佳结果。

- **init_params**: {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}, default=’kmeans’

初始化权重、均值和精度的方法。

| 方法             | 说明                   |
| ---------------- | ---------------------- |
| kmeans           | 使用 kmeans 初始化     |
| k-means++        | k-means++方法          |
| random           | 随机初始化             |
| random_from_data | 随机选择数据初始化均值 |   

- **weights_init**: array-like of shape (n_components, ), default=None

用户提供的初始权重。`None` 表示由 `init_params` 方法初始化权重。

- **means_init**: array-like of shape (n_components, n_features), default=None

用户提供的初始均值。`None` 表示由 `init_params` 方法初始化均值。

- **precisions_init**: array-like, default=None

用户提供的初始精度（协方差矩阵的逆）。`None` 表示由 `init_params` 方法初始化精度。shape 取决于使用的 `covariance_type`：

```python
(n_components,)                        if 'spherical',
(n_features, n_features)               if 'tied',
(n_components, n_features)             if 'diag',
(n_components, n_features, n_features) if 'full'
```

- **random_state**: int, RandomState instance or None, default=None

Controls the random seed given to the method chosen to initialize the parameters (see `init_params`). In addition, it controls the generation of random samples from the fitted distribution (see the method `sample`). Pass an int for reproducible output across multiple function calls. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

- **warm_start**: `bool`, default=False

如果 `warm_start` 为 True，则用上一次拟合的解作为下一次调用 `fit()` 的初始值。当对类似问题多次调用 `fit` 时，该方法可以加速收敛。此时，`n_init` 被忽略，只在开始调用时初始化一次。

**verbose**int, default=0

Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater than 1 then it prints also the log probability and the time needed for each step.

**verbose_interval**int, default=10

Number of iteration done before the next print.

## 属性

**weights_**array-like of shape (n_components,)

The weights of each mixture components.

**means_**array-like of shape (n_components, n_features)

The mean of each mixture component.

**covariances_**array-like

The covariance of each mixture component. The shape depends on `covariance_type`:

(n_components,)                        if 'spherical',
(n_features, n_features)               if 'tied',
(n_components, n_features)             if 'diag',
(n_components, n_features, n_features) if 'full'

**precisions_**array-like

The precision matrices for each component in the mixture. A precision matrix is the inverse of a covariance matrix. A covariance matrix is symmetric positive definite so the mixture of Gaussian can be equivalently parameterized by the precision matrices. Storing the precision matrices instead of the covariance matrices makes it more efficient to compute the log-likelihood of new samples at test time. The shape depends on `covariance_type`:

(n_components,)                        if 'spherical',
(n_features, n_features)               if 'tied',
(n_components, n_features)             if 'diag',
(n_components, n_features, n_features) if 'full'

**precisions_cholesky_**array-like

The cholesky decomposition of the precision matrices of each mixture component. A precision matrix is the inverse of a covariance matrix. A covariance matrix is symmetric positive definite so the mixture of Gaussian can be equivalently parameterized by the precision matrices. Storing the precision matrices instead of the covariance matrices makes it more efficient to compute the log-likelihood of new samples at test time. The shape depends on `covariance_type`:

(n_components,)                        if 'spherical',
(n_features, n_features)               if 'tied',
(n_components, n_features)             if 'diag',
(n_components, n_features, n_features) if 'full'

**converged_**bool

True when convergence was reached in fit(), False otherwise.

**n_iter_**int

Number of step used by the best fit of EM to reach the convergence.

**lower_bound_**float

Lower bound value on the log-likelihood (of the training data with respect to the model) of the best fit of EM.

**n_features_in_**int

Number of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

New in version 0.24.

**feature_names_in_**ndarray of shape (`n_features_in_`,)

Names of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when `X` has feature names that are all strings.



## 方法

### aic

```python
aic(X)
```

当前模型对输入 `X` 的 Akaike 信息准则。

参数 `X` 为输入样本数组，shape (n_samples, n_dimensions)。

返回 aic: float 值，越小越好。

### fit

```python
fit(X, y=None)
```

用 EM 算法估计模型参数。

该方法拟合模型 `n_init` 次，使用具有最大似然或最低下界的模型设置参数。每次拟合时，该方法迭代执行 E 步和 M 步 `max_iter` 次，直到似然值或 lower bound 的变化小于 `tol`，否则引发 `ConvergenceWarning`。

如果 `warm_start` 为 `True`，则忽略 `n_init`，并在第一次调用 `fit` 时执行一次初始化。在连续调用 `fit` 时，从上次拟合停止的地方重新开始拟合。

## 示例

