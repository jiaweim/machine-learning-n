## 概率编程

## 简介

对概率论和贝叶斯统计有了基本了解后，下面介绍如何使用计算工具建立概率模型。具体来说，就是使用 PyMC 进行概率编程：使用代码指定统计模型，然后 PyMC 为我们解决这些模型。

贝叶斯统计在概念上非常简单，即从已知 **数据（data）** 推导未知**参数（parameter）** 的性质。

虽然概念上简单，但全概率模型经常无法得到解析解。该问题导致贝叶斯方法在很多年只能应用于一些小众领域。计算机和**数值方法**的发展，使得现在理论上可以解决任何推理问题，极大改变了贝叶斯数据分析实践。这些数值方法可以看作通用推理引擎（universal inference engines）。自动化推理使得概率编程语言（probabilistic programming languages, PPL）迅速发展，从而将模型的创建和推理分离。在 PPL 框架中，用户只需要指定概率模型，推理由计算机自动完成。

PPL 使得数据科学从业者能够更少的时间和更少出错的方式构建更复杂的模型。

### 用 PyMC 的方式抛硬币

上一章介绍了抛硬币问题，下面演示如何用 PyMC 解决该问题。这里使用生成的数据，因此知道真实的 $\theta$ 值，记为 `theta_real`：

```python
np.random.seed(123)
trials = 4
theta_real = 0.35 # 实际试验中该参数未知
data = pz.Binomial(n=1, p=theta_real).rvs(trials)
data
```

```
array([1, 1, 0, 0], dtype=int64)
```

有了数据后，需要指定模型。即指定先验和似然。对似然，我们使用参数为 $n=1, p=\theta$ 的二项分布；对先验，则使用参数为 $\alpha=\beta=1$ 的 Beta 分布。该参数设置的 Beta 分布等价于区间 [0, 1] 上的均匀分布。该模型的数学表示：

$$
\theta \sim Beta(\alpha=1,\beta=1)
$$

$$
Y\sim Binomial(n=1,p=\theta)
$$

PyMC 实现：

```python
```

