# 最近邻

- [最近邻](#最近邻)
  - [简介](#简介)
  - [无监督最近邻](#无监督最近邻)
    - [找到最近的邻居](#找到最近的邻居)
    - [KDTree 和 BallTree](#kdtree-和-balltree)
  - [近邻分类](#近邻分类)
  - [近邻回归](#近邻回归)
  - [最近邻算法](#最近邻算法)
    - [暴力法](#暴力法)
    - [K-D Tree](#k-d-tree)
    - [Ball Tree](#ball-tree)
    - [算法选择](#算法选择)
    - [近邻算法的有效指标](#近邻算法的有效指标)
  - [参考](#参考)

***

## 简介

`sklearn.neighbors` 提供了基于近邻的监督和无监督学习算法。无监督最近邻是许多其它学习方法的基础，特别是流形（manifold）学习和谱（spectral）聚类。基于监督近邻的学习有两种形式：对带离散标签数据的分类，对带连续标签数据的回归。

近邻方法的基本原理是，找到与新数据点距离最近的指定数量样本，并根据这些样本预测标签。样本的数量可以是自定义常数（kNN），也可以根据点的局部密度动态变化（基于半径的近邻方法）。衡量距离的方法有很多，其中标准欧氏距离最常见。基于近邻的方法也称为非泛化机器学习方法，因为它们只是“记住”它的所有训练数据（可能被转换为快速索引结构，如 Ball Tree 或 KD Tree）。

尽管很简单，最近邻在大量的回归和分类问题上都很成功，包括手写数字和卫星图场景等。作为一种非参方法，它在决策边界不规则的分类问题中很成功。

`sklearn.neighbors` 中的类支持 NumPy 数组和 `scipy.sparse` 矩阵作为输入。对密集矩阵，支持大连可能的距离 metrics；对稀疏矩阵，支持搜索任意 Minkowski metrics。

有许多学习算法的核心都依赖于最近邻，如[核密度估计](https://scikit-learn.org/stable/modules/density.html#kernel-density)。

## 无监督最近邻

`NearestNeighbors` 实现无监督近邻算法，它为三种不同近邻算法提供统一接口：`BallTree`, `KDTree` 以及基于 `sklearn.metrics.pairwise` 的暴力算法。使用 `algorithm` 关键字参数设置算法，可选值为 `['auto', 'ball_tree', 'kd_tree', 'brute']`。默认值 `'auto'` 表示根据训练数据确定最佳方法。各个算法的优缺点后面会讨论。

> **WARNING**
> 对最近邻算法，当两个邻居 $k+1$ 和 $k$ 的距离相同但是标签不同，则结果取决于训练数据的排序。

### 找到最近的邻居

对查找两组数据的最近邻居这种简单任务，可以使用 `sklearn.neighbors` 中的无监督算法：

```python
>>> from sklearn.neighbors import NearestNeighbors
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
>>> distances, indices = nbrs.kneighbors(X)
>>> indices
array([[0, 1],
       [1, 0],
       [2, 1],
       [3, 4],
       [4, 3],
       [5, 4]]...)
>>> distances
array([[0.        , 1.        ],
       [0.        , 1.        ],
       [0.        , 1.41421356],
       [0.        , 1.        ],
       [0.        , 1.        ],
       [0.        , 1.41421356]])
```

由于查询数据与训练集一样，所以每个点的最近邻居为自身，距离为零。

也可以生成一个稀疏图，显示相邻点之间的连接：

```python
>>> nbrs.kneighbors_graph(X).toarray()
array([[1., 1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0.],
       [0., 0., 0., 1., 1., 0.],
       [0., 0., 0., 1., 1., 0.],
       [0., 0., 0., 0., 1., 1.]])
```

### KDTree 和 BallTree

可以直接使用 `KDTree` 和 `BallTree` 类来查找最近邻居。两个类的接口相同，下面展示 `KDTree`：

```python
>>> from sklearn.neighbors import KDTree
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> kdt = KDTree(X, leaf_size=30, metric='euclidean')
>>> kdt.query(X, k=2, return_distance=False)
array([[0, 1],
       [1, 0],
       [2, 1],
       [3, 4],
       [4, 3],
       [5, 4]]...)
```

在 DistanceMetric 类中列出了可用的距离指标。

## 近邻分类

近邻分类是基于实例（instance-based）的学习算法，而非泛化学习，即它它不构造通用的内部模型，而是简单地存储训练数据的实例。分类是根据每个点附近各种类别的个数投票决定的。

scikit-learn 实现了两种不同的最近邻分类算法：

- `KNeighborsClassifier` 实现了经典的 knn 算法；
- `RadiusNeighborsClassifier` 根据固定半径 $r$ 内邻居数量进行分类。

对 `KNeighborsClassifier` 算法，$k$ 值的最优值高度依赖于数据，一般来说 $k$ 值大能够抑制噪声的影响，但会使分类边界不明显。

在数据采样不均匀的情况下，基于半径的邻近分类算法 `RadiusNeighborsClassifier` 可能更好。由用户指定半径 $r$，在稀疏邻域中使用更少的邻居进行分类。对高维参数空间，由于维度诅咒该方法效果不佳。

基本的最近邻分类使用统一权重，即分配给查询点的类别是根据最近邻的简单多数投票得到的。在某些情况下，使用加权近邻效果更好，比如更近的邻居对拟合贡献更大。可以通过 `weights` 关键字设置加权近邻。默认值 `weights = 'uniform'` 表示给每个邻居相同权重，`weights = 'distance'` 分配的权重与查询点距离的导数成正比。用于也可以定义距离函数来自定义权重。

## 近邻回归

当标签为连续值时，可以使用近邻回归。分配给查询点的标签根据其最近邻居标签的平均值计算。

scikit-learn 提供了两种最近邻回归算法：

- `KNeighborsRegressor` 实现了基于 $k$ 个最近邻居的回归；
- `RadiusNeighborsRegressor` 实现了基于半径 $r$ 的回归。

另外还提供了加权最邻近，使用 `weights` 关键字设置：

- 默认 `weights = 'uniform'` 表示所有点权重相同；
- `weights = 'distance'` 表示权重与点距离的导数成正比；
- 也可以自定义距离函数。

![](2023-03-14-19-16-52.png)

在 F[ace completion with a multi-output estimators](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multioutput_face_completion.html) 中演示了如何输出多个邻居。其中，输入 $X$ 是人脸上半部分的像素，输出 $Y$ 是人脸下半部分的像素。

![](2023-03-14-19-19-21.png)

## 最近邻算法

### 暴力法

最近邻居的快速计算是机器学习中的一个活跃研究领域。其中最简单的方法是直接计算数据集中所有点之间的距离：对维度为 $D$ 的 $N$ 个样本，该方法的计算复杂度为 $O[DN^2]$。对小数据样本，暴力算法非常有效。但随着样本数的增加，暴力方法很快就不可行。在 `sklearn.neighbors` 的类中，使用关键字 `algorithm = 'brute'` 来指定暴力搜索算法，并使用 `sklearn.metrics.pairwise` 中的例程计算距离。

### K-D Tree

为了解决暴力算法计算效率低的问题，人们发明了许多基于树的数据结构。这些数据结构通过有效编码样本的距离信息来减少计算距离的次数。基本思想是，如果点 $A$ 和点 $B$ 非常远，而 $B$ 与 $C$ 很近，那么点 $A$ 和 $C$ 也很远，不需要显式计算 $A$ 和 $C$ 的距离。通过这种方式计算复杂度降到 $O[DNlog(N)]$。相对暴力法显著改进。

KD 树（K-dimensional tree 的缩写）是一个早期实现，将二维四叉树和三维十叉树推广到任意维度。KD 树为二叉树结构，沿数据轴递归地划分参数空间，将其划分为嵌套的正交各向异性区域。KD 树的构造非常快，因为划分只沿着数据轴执行，不需要计算 D 维距离。构造好 KD 树后，计算查询点最近邻只需要 $O[log(N)]$ 次距离计算。

虽然 KD 树对低维（$D < 20$）邻居搜索很快，当 $D$ 很大时就不再那么有效，这是所谓的维度诅咒的一种表现。

在 scikit-learn 中，KD 数搜索算法使用关键字参数 `algorithm = 'kd_tree'` 指定，并使用类 `KDTree` 进行计算。

### Ball Tree

为了解决 KD 树在高维上的低效问题，提出了 ball tree 数据结构。KD 树沿着笛卡尔轴划分数据，而 ball 树在一系列超球面（hyper-sphere）中划分数据。这使得 ball 树的构建成本比 KD 树高，但是得到数据结构在结构化数据特别高效，即时是非常高的维度。

ball 数使用质心 $C$ 和半径 $r$ 递归的将数据划分到 nodes，这样 node 中的每个点都位于由 $C$ 和 $r$ 定义的超球面内。对邻居搜索使用三角不等式来减少候选数据点的个数：

$$|x+y|\le |x|+|y|$$

通过这种方法，计算查询点和质心的距离足以用来判断该点与 node 中余下点距离的范围。由于 ball 树的球形几何结构，在高维数据上其性能优于 KD 树，不过实际性能依赖于训练数的结构。

scikit-learn 中使用 `algorithm = 'ball_tree'` 来设置 ball 树搜索算法，并使用 `BallTree` 执行计算。

### 算法选择

给定数据集的最佳算法取决于许多因素：

- 样本个数 $N$ (即 `n_samples`)以及维度 $D$ (即 `n_features`)
  - 暴力法时间复杂度为 $O[DN]$
  - 球树法时间复杂度为 $O[Dlog(N)]$
  - KD 树的时间复杂度难以精确描述，当 $D$ 很小（小于 20），时间复杂度近似为 $O[Dlog(N)]$，此时 KD 树很高效；当 D 很大，时间复杂度近似为 $O[DN]$，由于树结构的额外开销，此时 KD 树比暴力法还慢。

对于小型数据集（N 小于 30），$log(N)$ 与 $N$ 相当，此时暴力法比基于树的方法更有效。`KDTree` 和 `BallTree` 通过通过 *leaf size* 解决该问题：当样本数低于该阈值时切换到暴力算法。这使得样本较少时两个算法的效率接近暴力法。

- 数据结构：数据的内在维度或数据的稀疏性。内在维度指数据所在的流形的维度 $d\le D$，以线性或非线性嵌入参数空间。

### 近邻算法的有效指标

通过 `DistanceMetric` 类的文档查看可用指标。也可以通过 `valid_metric` 属性查看有效指标，例如：

```python
>>> from sklearn.neighbors import KDTree
>>> print(sorted(KDTree.valid_metrics))
['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']
```



## 参考

- https://scikit-learn.org/stable/modules/neighbors.html
