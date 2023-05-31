# 高斯混合模型

2023-05-31
****
## 概述

高斯混合模型（Gaussian mixture model, GMM）是一种概率模型，它假设样本是由几个参数未知的高斯分布的混合产生的。从单个高斯分布生成的所有样本的 cluster 为椭圆形。不同 cluster 的形状、大小、密度和方向都可能不同。我们可以把混合模型看作一种广义 k 均值聚类，它包含数据的协方差结构以及隐高斯分布的中心信息。

`sklearn.mixture` 实现高斯混合模型（Gaussian Mixture Model, GMM），支持对角线（diagonal）、球面（spherical）、捆绑（tied）和全协方差矩阵（full covariance matrices），抽样，以及根据样品进行建模。
![[Pasted image 20230524104329.png|450]]
> 双组分 GMM：数据点以及等概率线

scikit-learn 实现了不同的类来估计高斯混合模型，对应不同的估计策略。

## 高斯混合

### 功能介绍

GMM 有多种变体，`GaussianMixture` 类实现期望极大算法（EM）拟合高斯混合模型，是最简单的 GMM 形式。`GaussianMixture` 需要提前知道高斯分布个数 $k$。例如，对 iris 数据集：

```python
import sklearn  
from sklearn.datasets import load_iris 
from sklearn.mixture import GaussianMixture  
  
data = load_iris()  
X = data.data  
y = data.target
  
gm = GaussianMixture(n_components=3, n_init=10)  
gm.fit(X)
```

其中 `n_components` 就是高斯分布个数。另外：

- `GaussianMixture.weights_`  shape(3,1)为权重 $\Phi$ 值，有几个高斯分布，就有几个值；
- `GaussianMixture.means_` shape(3,4) 为高斯分布均值，后面的 4 对应 4 个特征
- `GaussianMixture.covariances_` shape(3,4,4)为协方差矩阵

检查算法是否收敛，以及迭代的次数：

```python
>>> gm.converged_
True
>>> gm.n_iter_
17
```

训练好模型，使用 `predict()` 对新样本硬分类，返回样本最可能归属的 cluster；使用 `predict_proba()` 预测新样本归属各个 cluster 的概率，即软分类。

```python
>>> gm.predict(X)
array([0, 0, 1, ..., 2, 2, 2])
>>> gm.predict_proba(X).round(3)
array([[0.977, 0. , 0.023],
		[0.983, 0.001, 0.016],
		[0. , 1. , 0. ],
		...,
		[0. , 0. , 1. ],
		[0. , 0. , 1. ],
		[0. , 0. , 1. ]])
```

另外，GMM 是生成模型，因此可以用它生成新的样本（生成的样本按照 cluster 索引排序），例如：

```python
>>> X_new, y_new = gm.sample(6)
>>> X_new
array([[-0.86944074, -0.32767626],
		[ 0.29836051, 0.28297011],
		[-2.8014927 , -0.09047309],
		[ 3.98203732, 1.49951491],
		[ 3.81677148, 0.53095244],
		[ 2.84104923, -0.73858639]])
>>> y_new
array([0, 0, 1, 2, 2, 2])
```

还可以使用 `score_samples()` 方法估计模型在任何位置的概率密度，返回在指定位置的概率密度函数（PDF）的 log 值。值越大，概率密度越高：

```python
>>> gm.score_samples(X).round(2)
array([-2.61, -3.57, -3.33, ..., -3.51, -4.4 , -3.81])
```

计算这些 score 值的指数，返回给定样本位置的 PDF 值。PDF 不是概率值，而是概率密度，概率密度并不在 0 到 1 之间，可以是任意正数。PDF 对指定区间积分，获得概率值。

它还可以为多变量模型绘制置信椭圆，以及根据贝叶斯信息准则（Bayesian Information Criterion, BIC）评估数据中的cluster 数。

### 协方差约束

当数据的维度很高，或 cluster 很多，或样本很少，EM 会难以收敛到最优解。此时可以通过限制算法需要学习的参数个数来降低任务难度。一种方法是限制 cluster 的形状和方向，这可以通过对协方差矩阵施加约束来实现。`covariance_type` 超参数指协方差类型，置具有以下可选值：

- "spherical"

所有 cluster 都必须是球形，但它们可以有不同的直径（即不同方差）。

- "diag"

cluster 可以是任何大小的任意椭圆形，但椭圆的轴必须平行于坐标轴（即协方差矩阵必须是对角的）。

- "tied"

所有 cluster 必须有相同的椭圆形状、大小和方向，即所有 cluster 共享相同的协方差矩阵。只是位置不同。

- "full"

每个 cluster 都可以采取任意形状、大小和方向，即不约束协方差矩阵。

下面是不同类型的图示：

![[Pasted image 20230524110232.png|450]]

```ad-note
训练 `GaussianMixture` 模型的计算复杂度取决于样本数 $m$，维度数 $n$，cluster 数 $k$，以及协方差约束。如果 `covariance_type` 为 "spherical" 或 "diag"，则计算复杂度为 $O(kmn)$；如果 `covariance_type` 为 "tied" 或 "full"，则计算复杂度为 $O(kmn^2+kn^3)$，因此无法处理太多的特征。
```

### GaussianMixture 的优缺点

**优点：**

- 速度：它是学习混合模型最快的算法
- 不可知论：该算法只最大化可能性，所以没有均值向 0 的 bias，也不会使 cluster 大小偏向于特定结构

**缺点：**

- 奇异点：当 mixture 的数据点不足，会难以估计协方差矩阵。
- 组分数：该算法会使用它可以访问的所有组分，需要保留数据或信息理论标准在缺乏外部线索的情况决定使用多少组分。

### EM 估计算法

从无标记数据中学习混合高斯模型的主要难点在于不知道数据点来自哪个潜在分类（如果类别已知，为每个分组点拟合一个单独的高斯分布太容易了）。期望最大化（EM）是一个成熟的统计方法，通过迭代来解决该问题。首先，随机假设分类（随机在数据点居中，或从 k-means 学习，或只是在原点周围正态分布），计算每个点由每个分类生成的概率；然后，调整参数以最大化给定这些分配的概率。重复这个过程，直到收敛到局部最优。

### 选择初始化方法

有四种初始化方法（以及用户自定义输入的均值）来为模型分类生成初始中心：

- **k-means (默认)**

采用传统的 k 均值聚类算法。相对其它初始化方法，这个计算量更大。

- **k-means++**

k 均值聚类算法。从数据中随机选择第一个中心，随后从数据的加权分布中选择其他中心，优先选择与现有中心远的数据。k-means++ 是 k-means 的默认初始化方法，比运行完整的 k-means 快，但对于包含许多类别的大型数据集依然需要大量时间。

- **random_from_data**

从输入数据中随机挑选数据点作为初始中心。该方法非常快，但如果选择的点太近，会难以收敛。

- **random**

从所有数据的均值添加扰动来选择初始中心。该方法简单，但是收敛时间更长。

![[Pasted image 20230529205147.png|450]]
## 选择 cluster 数

和 k-means 一样，`GaussianMixture` 需要手动指定 cluster 数。对 k-means，可以用 inertia 或 silhouette score 选择合适的 cluster 数。但是，这些度量对高斯混合都不可靠，相反，可以尝试找到最小化理论信息准则（theoretical information criterion）的模型，如贝叶斯信息准则（Bayesian Information Criterion, BIC）和 Akaike Information Criterion (AIC)。

$$BIC=log(m)p-2log(\widehat{L})$$

$$AIC=2p-2log(\widehat{L})$$
其中：

- m 是样本数
- p 模型学习的参数个数
- $\widehat{L}$ 是模型似然函数的最大值。

BIC 和 AIC 都会惩罚参数更多的模型（如更多 cluster），奖励与数据拟合更好的模型。大多时候两者获得的模型相同。如果出现不同，BIC 选择的模型一般比 AIC 选择的模型更简单（参数更少），但对数据的拟合要差一点（对大型数据集更是如此）。

使用 `bic()` 和 `aic()` 计算 BIC 和 AIC：

```python
>>> gm.bic(X)
8189.747000497186
>>> gm.aic(X)
8102.521720382148
```

下图是不同 k 值对应的 BIC 和 AIC 值，可以看到，AIC 和 BIC 都是在 $k=3$ 时最小，所以 cluster 设置 3 大概率最合适。

![[Pasted image 20230531110712.png]]
> 不同 cluster 数 k 对应的 AIC 和 BIC 值

手动挑选最合适的协方差类型和 cluster 数：

```python
results = []  
covariance_types = ['full', 'tied', 'diag', 'spherical']  
for n_components in range(1, 9):  
    for covariance_type in covariance_types:  
        mclust = GaussianMixture(n_components=n_components,  
                                 warm_start=True,  
                                 covariance_type=covariance_type)  
        mclust.fit(df)  
        results.append({  
            'bic': mclust.bic(df),  
            'n_components': n_components,  
            'covariance_type': covariance_type,  
        })
```

> `warm_start` 参数表示是否重用上一次拟合结果，这可以加快下一次拟合的收敛速度。

## 似然函数

概率（probability）和似然（likelihood）在日常中往往互换使用，但在统计学中它们的含义截然不同。对具有参数 $\theta$ 的统计模型：

- 概率用于描述已知参数 $\theta$ ，未来出现结果 x 的可能性
- 似然用于描述已知结果 x 时，一组特定参数值 $\theta$ 的可能性

例如，设有一个 1D 混合模型，两个高斯分布的中心分别为 -4 和 +1。为了简单起见，设该模型只有一个 $\theta$ 参数控制两个分布的标准差。下图（左上）显示模型 $f(x;\theta)$ 与 x 和 $\theta$ 关系。为了估计 x 的概率分布，需要设置模型参数 $\theta$。例如，如果设置 $\theta$ 为 1.3 （水平线），就得到概率密度函数 $f(x;\theta=1.3)$ （左下）。假设你想估计 x 在 -2 到 2+ 之间的概率，则需要计算 PDF 在该范围的积分（左下蓝色区域）。但是，如果你不知道 $\theta$ ，而是知道单个样本 $x=2.5$（左上垂直蓝线）？此时得到的是似然函数 $L(\theta|x=2.5)=f(x=2.5;\theta)$（右上）。

![[Pasted image 20230531084228.png]]
> 模型参数函数（左上），PDF（左下），似然函数（右上），log 似然函数（右下）

简而言之，PDF 是关于 $x$ 的函数（$\theta$ 固定），而似然函数是关于 $\theta$ 的函数（$x$ 固定）。要理解似然函数不是概率分布：如果将概率分布对所有 x 的可能值积分，得到 1；如果将似然函数对所有可能的 $\theta$ 积分，结果可能是任何正数。

给定数据集 X，一个常见任务是估计模型参数的最优值，该功能通过最大化似然函数的值实现。在上例中，如果发现对样本 $x=2.5$，$\theta$ 的最大似然估计（maximum likelihood estimate, MLE）值为 $\hat{\theta}=1.5$ 。如果存在关于 $\theta$ 的先验概率分布 $g$，则可以通过最大化 $L(\theta|x)g(\theta)$ 实现，这被称为最大后验估计（maximum a-posteriori, MAP）。由于 MAP 约束了参数值，因此可以将其视为 MLE 的正则化版本。

最大化似然函数等价于最大化它的对数（右下图）。对数函数是一个严格的递增函数，所以如果 $\theta$ 使对数似然最大化，那么一定也是似然最大化。事实证明，最大化对数似然函数更容易。例如，如果你有$x^{(1)}$ 到 $x^{(m)}$ 的独立样本，需要找到 $\theta$ 值使得单个似然函数的乘积最大化，这等价于最大化对数似然函数的和（不是乘积），即 $log(ab)+log(a)+log(b)$。

获得最大化似然函数的 $\theta$ 的值 $\hat{\theta}$ 后，就可以计算 $\hat{L}=L(\hat{\theta},X)$，就能计算 AIC 和 BIC 值。

## 贝叶斯高斯混合模型

除了手动搜索最合适的 cluster 数，也可以使用 `BayesianGaussianMixture` 类，它会自动将不必要的 cluster 的权重设置为 0。将 cluster 数 `n_components` 设置为一个大于最优 cluster 数的值（需要对问题有所了解），算法会自动删除不必要的 cluster。例如，我们将 cluster 数设置为 10：

```python
>>> from sklearn.mixture import BayesianGaussianMixture
>>> bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
>>> bgm.fit(X)
>>> bgm.weights_.round(2)
array([0.4 , 0.21, 0.4 , 0. , 0. , 0. , 0. , 0. , 0., 0. ])
```

完美！贝叶斯 GMM 自动检测到只有三个 cluster。

`BayesianGaussianMixture` 通过变分推理算法实现高斯混合模型。

### 变分推理估计算法

变分推理是 EM 的一个扩展，其原理与 EM 相同（都是迭代算法，计算每个 mixture 生成每个点的概率，将 mixture 拟合到这些点），但是变分方法通过集成先验分布的信息增加了正则化，从而避免了 EM 中常出现的奇异性问题。该方法推理明显变慢，但也没慢到无法使用的程度。

由于贝叶斯的性质，变分算法相比 EM 需要更多超参数，其中 `weight_concentration_prior` 参数最重要。较小的 `weight_concentration_prior` 使得模型将权重放在少数组分上，余下 cluster 的权重接近 0；而较大值使得 mixture 中有更多组分。

`BayesianGaussianMixture` 类的参数实现提出了两种先验权重分布：服从 Dirichlet 分布的有限混合模型和服从 Dirichlet 过程的无限混合模型。在实践中，Dirichlet 过程推理算法是近似的，使用固定最大 cluster 数的截断分布。实际使用的 cluster 数取决于数据。



下图对比了不同 `weight_concentration_prior_type` 和不同 `weight_concentration_prior` 的结果。可以看到，`weight_concentration_prior` 参数对获得的有效组分数影响较大。而且对 "dirichlet_distribution" 类型，`weight_concentration_prior` 的值越大，权重值分布越均匀。

![[Pasted image 20230529210729.png|600]]

![[Pasted image 20230529210714.png|600]]

下面比较了具有固定组分数的高斯混合模型和具有 Dirichlet 过程的变分高斯混合模型。这里，经典高斯混合在包含 2 个 cluster 的数据集上拟合了 5 个分量。可以看到，具有 Dirichlet 过程的先验变分高斯混合能够自动推理出 2 个组分，而高斯混合的组分数只能由用户指定。在这里，用户指定 `n_components=5` 与真实的分布不匹配。注意，在少数情况下，具有 Dirichlet 过程先验的变分高斯混合模型会采取保守策略，只拟合出一个分量。

![[Pasted image 20230529211554.png|600]]
下图是对一个高斯混合不能很好描述的数据集的拟合。调整 `BayesianGaussianMixture` 的 `weight_concentration_prior` 参数控制拟合该数据的组件数。

## 应用示例

高斯混合模型在椭圆形状的 cluster 工作很好，对其它差异很大的 cluster 效果不好。例如，用贝叶斯 GMM 对 moons 数据集聚类。

![[Pasted image 20230531112746.png]]

贝叶斯 GMM 拟合了很多椭圆，找到了 8 个不同的 cluster，而不是实际的两个。密度估计效果还行，所以这个模型或许可以用于异常检测，但它没能识别出两颗卫星。

### 异常检测

使用 GMM 做异常检测（anomaly detection）非常简单：任何位于低密度区的样本都可以认为是异常的。因此，首先要定义密度阈值。例如，在做产品检测时，将不良品的比例控制在 2% ，因此设置的密度阈值就要保证 2% 的样本位于阈值以下的低密度区。

如果你发现有太多假阳性（即合格产品被标记为不良品），则可以降低阈值；相反，乳沟发现太多假阴性（不良品被标记为合格），则需要提高阈值。这就是 precision/recall 权衡。

例如，将密度的第 4 个百分位数作为阈值来识别异常值，即大约 4% 的样本将被标记为异常：

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

新颖性检测（novelty detection）和异常检测类似，差别在于：新颖性检测要求模型在"干净"的数据集上训练，即训练集不包含异常值，而异常检测不做该加假设。事实上，异常检测常用于清理数据集。

```ad-tip
GMM 试图拟合所有数据，包括异常值；如果训练集包含太多异常值，会使模型产生 bias，从而错误地将一些异常值（outlier）看作正常的。如果发生该情况，可以先拟合一次模型，使用它来检测并删除极端的异常值，删除这些异常值后，再次拟合模型。另一种方法是使用 robust 的协方差估计方法，如 `EllipticEnvelope` 类。
```

### 其它异常检测算法

Scikit-Learn 实现了其它专门用于异常检测或新颖检测的算法，包括：

- **Fast-MCD** (Minimum Covariance Determinant)

该算法由 `EllipticEnvelope` 类实现，用于异常检测，特别是数据集清理方面。它假设正常样本（inlier）由单个高斯分布生成，同时假设数据集种的离群值（outlier）不是由这个高斯分布产生。在估计高斯分布参数（即围绕 inlier 的椭圆的形状）时，它会避开那些可能得 outliers。该技术可以更好地估计椭圆形状，从而能更好地识别 outliers。

- **Isolation forest**

这也是一个异常检测算法，特别是对高维数据集中。该算法构建一个随机森林，每个决策树随机生长：在每个 node，随机选择一个 feature，然后选择一个随机阈值（feature 最小和最大值之间）将数据集分为两部分。数据集以这种方式逐渐被分割成小块，直到将每个样本都分开。异常样本通常与其他样本相隔较远，因此相比正常样本，它们只需要少数步骤就能被分割开。

- **Local outlier factor** (LOF)

该算法也适用于异常值检测。该算法比较样本与其邻居样本的概率密度，异常值与它最近的 $k$ 个邻居的概率密度通常不同。

- **One-class SVM**

该算法适合新颖性检测。SVM 通过将所有样本映射到高维空间，然后使用线性 SVM 分类器进行分类。因为只有一类样本，所以 one-class SVM 是在高维空间将样本与原点分离。在原始空间中，等价于找到一个包含所有样本的小区域。如果新样本不在此区域，就为异常。它效果很好，特别是高维数据集，但是不能扩展到大型数据集。

### 两组分高斯混合模型

使用不同中心和方差的两个高斯分布生成数据，绘制混合高斯分布的密度估计。

```python
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.colors import LogNorm  
from sklearn import mixture  
  
n_samples = 300  
  
np.random.seed(0)  
  
# 生成以 (20,20) 为中心的球面数据  
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])  
  
# 生成以 (0,0) 为中心的拉伸高斯数据  
C = np.array(([[0.0, -0.7], [3.5, 0.7]]))  
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)  
  
# 合并两个数据集  
X_train = np.vstack([shifted_gaussian, stretched_gaussian])  
  
# 拟合 2 组分混合高斯模型  
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")  
clf.fit(X_train)  
  
# 以等高线的形式显示模型预测的分数  
x = np.linspace(-20.0, 30.0)  
y = np.linspace(-20.0, 40.0)  
X, Y = np.meshgrid(x, y)  
XX = np.array([X.ravel(), Y.ravel()]).T  
Z = -clf.score_samples(XX)  
Z = Z.reshape(X.shape)  
  
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))  
CB = plt.colorbar(CS, shrink=0.8, extend="both")  
  
plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)  
plt.title("Negative log-likelihood predicted by a GMM")  
plt.axis("tight")  
plt.show()
```

![[Pasted image 20230529222230.png]]

### 贝叶斯 GMM 的先验浓度类型

使用 `BayesianGaussianMixture` 类拟合一个 toy 数据集（三个高斯分布混合），


### GMM 协方差

这里将 GMM 聚类获得的 cluster 与数据集中实际的类别进行比较。为了使该比较有效，用训练集中的类的均值初始化高斯均值。

在 iris 数据集上，使用不同 GMM 协方差类型进行聚类，绘制训练和测试集的预测标签。尽管我们期望 full 协方差的性能最好，但它容易在小数据集上过拟合，不能很好地泛化到测试数据。

在图中，训练数据以点显示，测试数据用 x 显示。iris 数据集有四维，这个只显示了前两个维度，因此有些点在其他维度是分开的。

## 总结

GMM 作为一个基于模型的聚类技术，对数据的分布进行了假设。因此，GMM 聚类效果依赖该该假设，即数据是否服从多变量混合高斯分布。

另外，GMM 的计算复杂度高，甚至高于分层聚类，因此难以扩展到大型数据集。

## 参考

- https://scikit-learn.org/stable/modules/mixture.html
- 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3ed》，Aurélien Géron
- 《Practical Statistics for Data Scientists， 2ed》，Peter Bruce, Andrew Bruce & Peter Gedeck