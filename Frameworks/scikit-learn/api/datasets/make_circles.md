# sklearn.datasets.make_circles

2023-05-31
****
## 简介

```python
sklearn.datasets.make_circles(n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8)
```

在 2D 空间中生成一个大圆套小圆的数据集。

一个简单的 toy 数据集，用于可视化聚类和回归算法。

## 参数

- **n_samples**: `int` or `tuple` of shape (2,), `dtype`=int, default=100

如果为 `int` 值，则是生成的数据点数。奇数时，内圈比外圈多一个点。
如果为双元素 tuple，则分别是外圈和内圈的点数。

- **shuffle**: `bool`, default=True

是否 shuffle 样本。

- **noise**: float, default=None

加入数据的高斯噪声的标准差。

- **random_state**: `int`, `RandomState` instance or None, default=None

用于确定 shuffle 和噪声的随机数。传入相同 `int` 可重现结果。

- **factor**: float, default=.8

内圈和外圈的比例，范围 `(0,1)`。

## 返回

- **X**: ndarray of shape (n_samples, 2)

生成的样本。

- **y**: ndarray of shape (n_samples,)

样本所属类别的整数标签（0 或 1）。
