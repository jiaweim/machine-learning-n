# sklearn.datasets.make_moons

2023-05-31
****
## 简介

```python
sklearn.datasets.make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None)
```

生层两个相互交错的半圆。

一个简单的 toy 数据集，用于可视化聚类和分类算法。

## 参数

- **n_samples**: `int` or `tuple` of shape (2,), dtype=int, default=100

如果为 `int` 值，则是生成的数据点数。
如果为双元素 tuple，则分别是两个 moons 的点数。

- **shuffle**: bool, default=True

是否 shuffle 样本。

- **noise**: float, default=None

加入数据的高斯噪声的标准差。

- **random_state**: int, RandomState instance or None, default=None

用于确定 shuffle 和噪声的随机数。传入相同 `int` 可重现结果。

## 返回

- **X**: ndarray of shape (n_samples, 2)

生成的样本。

- **y**: ndarray of shape (n_samples,)

样本所属类别的整数标签（0 或 1）。
