# 分类

## 简介

下面介绍如下内容：

- 介绍流行的分类算法，如逻辑回归、支持向量机、决策树和 k-最近邻
- 使用 scikit-learn 解释和演示机器学习算法
- 讨论不同分类器的优缺点

## 选择分类算法

每种算法都有各自的优缺点和基本假设，为特定任务选择合适的分类算法需要实践和经验。没有哪个分类器在所有情况都是最好的。在实践中，建议比较几种不同的机器学习算法的性能，以选择针对特定问题的最佳模型。最佳算法受多个因素影响，包括特征和样本数量、数据集中噪音量，以及是否线性可分。

分类器的性能（计算性能和预测性能）在很大程度上取决于训练集样本。训练监督学习模型的主要步骤：

1. 选择特征，收集标记训练样本
2. 选择评估性能指标
3. 选择学习算法并训练模型
4. 评估模型性能
5. 修改算法设置，调优模型

## 训练感知机

使用 Iris 数据集训练感知机模型，载入 iris 数据：

```python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print("Class labels:", np.unique(y))
```

```txt
Class labels: [0 1 2]
```

为了方便可视化，这里只用于 petal length 和 petal width 两个特征。

使用 `train_test_split` 函数将样本拆分为训练集和测试集：

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
```

其中 30% 用作测试数据（45 个样本），70% 用作训练数据（105 个样本）。

`train_test_split` 在拆分数据前在内部对数据集进行了洗牌，设置随机种子 `random_state=1` 是为了保证本示例的可重复性。

`stratify=y` 表示使 `train_test_split` 返回的训练集和测试集具有与输入数据集相同类标签的比例。可以用 `np.bincount` 函数统计不同标签个数：

```python
print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))
```

```txt
Labels counts in y: [50 50 50]
Labels counts in y_train: [35 35 35]
Labels counts in y_test: [15 15 15]
```

许多机器学习算法和优化算法为了获得最佳性能，需要对特征进行缩放。下面使用 `StandardScaler` 类标准化特征：

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

`StandardScaler.fit` 方法从训练数据计算每个特征的均值 $\mu$ 和标准差 $\sigma$。调用 `transform` 将训练数据和测试数据标准化，注意测试数据要使用与训练数据一样的标准化参数。

标准化数据后，开始训练感知机模型。scikit-learn 中大多数算法默认通过 one-versus-rest (OvR)支持多分类。下面训练感知机 `Perceptron` 模型：

```python
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
```

这里，参数 `eta0` 为学习率。找到一个合适的学习率需要多尝试。如果学习率过大，可能会跳过全局最小损失值；如果学习率太小，则需要更长的训练时间，使得学习变慢。这里依然用 `random_state` 参数来确保每个 epoch 之后训练集洗牌的可重复性。

训练好模型后，可以用 `predict` 预测新数据：

```python
y_pred = ppn.predict(X_test_std)
print("Misclassified examples: %d" % (y_test != y_pred).sum())
```

```txt
Misclassified examples: 1
```

执行代码，可能发现 45 个测试样本分类错了一个。因此在测试数据上的错误分类率为 $\frac{1}{45}\approx 0.022$，即 2.2% 的错误率。

scikit-learn 在 `metrics` 模块中实现了大量性能指标。例如，在测试数据集上计算感知机的分类精度：

```python
from sklearn.metrics import accuracy_score

print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
```

```txt
Accuracy: 0.978
```

`y_test` 是真实类别标签，`y_pred` 是预测类别。另外，scikit-learn 的每个分类器都有一个 `score` 方法，该方法合并 `predict` 和 `accuracy_score` 方法计算分类器的预测精度：

```python
print("Accuracy: %.3f" % ppn.score(X_test_std, y_test))
```

```txt
Accuracy: 0.978
```

下面使用 `plot_decision_regions` 函数绘制模型的决策区域，并可视化它如何分离不同花样本。

## 参考

- Machine Learning with PyTorch and Scikit-Learn, Sebastian Raschka & Yuxi Liu & Vahiid Mirjalili
- https://scikit-learn.org/stable/supervised_learning.html
