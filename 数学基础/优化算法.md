# 优化算法

- [优化算法](#优化算法)
  - [概述](#概述)
  - [参数与超参数](#参数与超参数)
    - [验证数据](#验证数据)
    - [超参数优化](#超参数优化)
  - [SGD](#sgd)
    - [SGD 缺点](#sgd-缺点)
  - [Momentum](#momentum)
    - [Momentum Python 实现](#momentum-python-实现)
  - [AdaGrad](#adagrad)
    - [AdaGrad Python 实现](#adagrad-python-实现)
  - [Adam](#adam)
  - [优化算法选择](#优化算法选择)

2021-06-04, 09:14
***

## 概述

在确定了训练集D、假设空间F以及学习准则后，如果找到最优的模型 $f(x,\theta^*)$就成了一个最优化（Optimization）问题。机器学习的训练过程其实就是最优化问题的求解过程。

## 参数与超参数

在机器学习中，优化可以分为参数优化和超参数优化。模型 $f(x;\theta)$ 中的θ 称为模型的参数，可以通过优化算法进行学习。除了可学习的参数 θ 之外，还有一类参数是用来定义模型结构或优化策略的，这类参数叫做**超参数**（hyper parameter）。

### 验证数据

不能使用测试数据评估超参数的性能，因为如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合。换句话说，用测试数据确认超参数的好坏，就会导致超参数的值被调整为只拟合测试数据。这样就可能得到不能拟合其它数据、泛化能力差的模型。

因此，调整超参数时，必须使用超参数专用的数据。用于调整超参数的数据，一般称为验证数据（validation data）。

> 训练数据用于参数（权重和偏置）的学习，验证数据用于超参数的性能评估。为了确认泛化能力，要在最后使用（比较理想的是只使用一次）测试数。

### 超参数优化

进行超参数优化时，要逐渐缩小超参数合适值的选择范围。即一开始大致设定一个范围，从这个范围中随机选出一个超参数（采样），用这个采样到的值进行识别精度的评估；然后多次重复该操作，观察识别精度的结果，根据这个结果缩小超参数的范围。

有报告显示，在进行神经网络的超参数的最优化时，与网格搜索等有规律的搜索相比，随机采样的搜索方式效果更好。这是因为在多个超参数时，各个超参数对最终的识别精度的影响程度不同。

超参数的范围只要大致指定就可以了。这个大致是指 从 $10^{-3}$ 到 $10^3$ 这样，以10的阶乘的尺度指定范围。

在超参数的优化中，要注意的是深度学习需要很长时间（比如，几天或几周）。因此，在超参数的搜索中，需要尽早放弃那些不符合逻辑的超参数。减少学习的 epoch，缩短一次评估所需的时间。

简而言之，超参数的优化步骤为：

1. 设定超参数的范围；
2. 从设定的超参数范围中随机采用；
3. 使用步骤 1 中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将 epoch 设置得很小）；
4. 重复步骤1和2（比如 100 次），根据识别精度的结果，缩小超参数的范围。

该超参数的最优化方法是实践性的方法，在超参数的最优化中，如果需要更精炼的方法，可以使用贝叶斯最优化（Bayesian optimization）。贝叶斯最优化运用以贝叶斯定理为中心的数学理论，能够更加严密、高效地进行最优化。

## SGD

随机梯度下降法（stochastic gradient descent）。用数学式可以将 SGD 表示为：

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W} \tag{1}
$$

$W$ 为权重参数，$\frac{\partial L}{\partial W}$ 为损失函数关于 $W$ 的梯度。$\eta$ 表示学习率。

Python 实现：

```py
class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """
        更新参数。
        :param params: 权重参数
        :param grads: 梯度
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

### SGD 缺点

SGD 实现简单，但是在解决某些问题时没有效率。如果函数的形状为非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。SGD 低效的根本原因是，梯度的方向没有指向最小值的方向。

## Momentum

Momentum 表示动量，用学术式表示 Momentum 方法：

$$
v \leftarrow \alpha v-\eta\frac{\partial L}{\partial W} \tag{2}
$$

$$
W \leftarrow W+v \tag{3}
$$

这里 $W$ 是要更新的权重参数，$\frac{\partial L}{\partial W}$ 表示损失函数关于 $W$ 的梯度，$\eta$ 表示学习率。式（2）表示物体在梯度方向受力，物体的速度在力的作用下改变。

变量 $v$ 对应物理上的速度。$\alpha v$ 表示在物理不受任何力时，使物理逐渐减速，对应物理上的摩擦力或空气阻力。$\alpha$ 一般设置为 0.9 之类的值。

### Momentum Python 实现

```py
class Momentum:
    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

## AdaGrad

AdaGrad 为参数的每个元素适当地调整学习率，AdaGrad 的 Ada 为 Adaptive 缩写，即适当的调整学习率。

$$
h \leftarrow h + \frac{\partial L}{\partial W} \bigodot \frac{\partial L}{\partial W}
$$

$$
W \leftarrow W - \eta\frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}
$$

说明：

- $W$ 表示要更新的权重参数
- $\frac{\partial L}{\partial W}$ 表示损失函数关于 $W$ 的梯度
- $\eta$ 表示学习率
- $h$ 保存了以前所有梯度的平方和（$\bigodot$ 表示矩阵元素的乘法）

在更新参数时，通过乘以 $\frac{1}{\sqrt{h}}$ 可以调整学习的尺度。参数变动较大（被大幅度更新）的元素学习率将变小，所以可以按参数的元素进行学习率的衰减，使变动大的参数的学习率逐渐减小。

AdaGrad 记录过去所有梯度的平方和，因此，学习越深入，更新的幅度就越小。

### AdaGrad Python 实现

```py
class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

## Adam

Adam 结合 Momentum 和 AdaGrad方法，实现参数空间的高效搜索。

## 优化算法选择

经典的 SGD 和高度优化的 Adam 比较推荐。一般而言，与 SGD 相比，Momentum、AdaGrad 和 Adam 学习得更快，有时候最终的识别精度也更高。

