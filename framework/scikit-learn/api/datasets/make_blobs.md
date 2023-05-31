# sklearn.datasets.make_blobs

2023-05-31
****
## 简介

```python
sklearn.datasets.make_blobs(n_samples=100, 
							n_features=2, *, 
							centers=None, 
							cluster_std=1.0, 
							center_box=(-10.0, 10.0), 
							shuffle=True, 
							random_state=None, 
							return_centers=False)
```

`make_blobs()` 从一个特殊的高斯混合模型中抽样，生成用于聚类的各向同性混合高斯分布。

包含 k 个 cluster 的高斯混合模型的概率密度为：

$$p(x)=\sum^{k}_{i=1}\pi_i N(\mu_i, \sum_i)$$
其中 $\pi_i\ge 0$ 是每个 cluster 的权重，$\sum^{k}_{i=1}\pi_i=1$；$\mu_i$ 是 cluster 中心，$\sum_i$ 是 cluster 协方差。$N(\mu_i,\sum_i)$ 是均值为 $\mu_i$ 协方差为 $\sum_i$ 的多元高斯分布。

`make_blobs()` 函数中，每个 cluster 被抽样的概率相同，即 $\pi_i=\frac{1}{k}$；cluster 的中心可以手动指定，也可以设置 `centers=2` 个数后随机生成。各向同性（isotropic）指协方差矩阵为对角矩阵：

$$\sum_i=\begin{bmatrix}
\sigma^2_i & 0 \\
0 & \sigma^2_i
\end{bmatrix}$$
其中 $\sigma_i$ 为标准差。默认情况下，所有 cluster 的标准差相同。如果只有一个中心，那么高斯混合模型与高斯模型没有差别。

## 参数

- **n_samples**: int or array-like, default=100

如果为 `int`，则是生成的数据点数，在所有 cluster 平均分配。
如果为 array-like，则分别指定每个 cluster 的样本数。

- **n_features**: int, default=2

每个样本的 feature 数。

- **centers**: int or ndarray of shape (n_centers, n_features), default=None

中心数，或者固定的中心位置。如果 `n_samples` 是 int，而 `centers` 是 `None`，则生成 3 个中心。如果 `n_samples` 是 array-like，则 `centers` 要么为 `None`，要么是长度等于 `n_samples` 的数组。

- **cluster_std**: float or array-like of float, default=1.0

cluster 的标准差。

- **center_box**: tuple of float (min, max), default=(-10.0, 10.0)

在随机生成中心时，每个 cluster 中心的边界。

- **shuffle**: bool, default=True

shuffle 样本。

- **random_state**: int, RandomState instance or None, default=None

设置随机数生成器。

- **return_centers**: bool, default=False

是否返回每个 cluster 的中心。

## 返回

- **X**: ndarray of shape (n_samples, n_features)

生成的样本。

- **y**: ndarray of shape (n_samples,)

样本所属类别的整数标签（0 或 1）。

- **centers**: ndarray of shape (n_centers, n_features)

每个 cluster 的中心。当  `return_centers=True` 时返回。

## 示例

```python
>>> from sklearn.datasets import make_blobs
>>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
>>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
...                   random_state=0)
>>> print(X.shape)
(10, 2)
>>> y
array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])
```
