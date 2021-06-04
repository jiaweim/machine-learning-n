# 激活函数

- [激活函数](#激活函数)
  - [简介](#简介)
  - [阶跃函数](#阶跃函数)
  - [Sigmoid 函数](#sigmoid-函数)
    - [Logistic 函数](#logistic-函数)
    - [Tanh 函数](#tanh-函数)
    - [Hard-Logistic 和 Hard-Tanh](#hard-logistic-和-hard-tanh)
  - [ReLU](#relu)

2021-05-26, 15:47
@Jiawei Mao
***

## 简介

激活函数在神经元中非常重要，为了增强网络的表示能力和学习能力，激活函数需要具备以下几点性质：

1. 连续并可导（允许少数点不可导）的**非线性函数**。可导的激活函数可以直接利用数值优化的方法来学习网络参数。
2. 激活函数及其导函数要尽可能简单，有利于提高网络计算效率。
3. 激活函数的导函数的值要在一个合适的区间内，否则会影响训练的效率和稳定性。

感知机和神经网络的主要差别就在于激活函数。

|激活函数|函数|应用|
|---|---|---|
|单位阶跃函数（Unit step）|$$|Perceptron|

## 阶跃函数

$$
h(x)=\begin{cases}
    0 \quad(x \le 0) \\
    1 \quad(x > 0)
\end{cases}
$$

阶跃函数可以看作一个特殊的激活函数，根据输入是否达到阈值，输出 0 或者 1，感知机使用该激活函数。

Python 实现：

```py
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
```

## Sigmoid 函数

Sigmoid 型函数指一类 S 型曲线函数，为两端饱和函数。常用的 Sigmoid 型函数有 Logistic 函数和 Tanh 函数。

> 对函数 f(x)，若 $x\rightarrow -\infty$ 时，其导数 $f'(x) \rightarrow 0$，则称为**左饱和**。若 $x\rightarrow +\infty$时，其导数 $f'(x) \rightarrow 0$，则称其为**右饱和**。当同时满足左、右饱和时，就称为**两端饱和**。

![sigmoid 激活函数](images/2021-05-26-15-54-55.png)

### Logistic 函数

Logistic 函数定义如下：

$$
\sigma (x)= \frac{1}{1+exp(-x)}
$$

特征：

- 当输入值在 0 附近时，Sigmoid 型函数近似为线性函数
- 当输入值靠近两端时，对输入进行抑制，输入越小，越接近0；输入越大，越接近 1

Logistic 函数的性质使得对应神经元具有以下特征：

- 输出可以直接看作概率分布，使得神经网络可以很好地和统计学习模型结合；
- 可以看作一个软性门（soft gate），用来控制其它 神经元信输出信息的数量。

Python 实现：

```py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Tanh 函数

Tanh 也是一种 Sigmoid 函数，其定义为：

$$
tanh(x)=\frac{exp(x)-exp(-x)}{exp(x)+exp(-x)}
$$

Tanh 函数可以看作放大并平移的 Logistic 函数，其值域为 (-1,1)。

Tanh 函数的树池是零中心化的（zero-centered），而 Logistic 函数的输出恒大于 0.非零中心化的输出会使得最后一层神经元的输入发生偏置转移（bias shift），并进一步使得梯度下降的收敛速度变慢。

### Hard-Logistic 和 Hard-Tanh

Logistic 函数和 Tanh 函数都是 Sigmoid 型函数，具有饱和性，但是计算开销较大。因为这两个函数都是在中间（0附近）近似线性，两端饱和，因此可以通过分段函数来近似。

以 Logistic 函数 $\sigma(x)$ 为例，其导数 $\sigma'(x)=\sigma(x)(1-\sigma(x))$。Logistic 函数在 0 附近的一阶泰勒展开为：



## ReLU

ReLU（Rectified Linear Unit, 修正线性单元）是目前深度神经网络中最常使用的激活函数。其定义为：

$$
\begin{aligned}
\sigma(x)&=\begin{cases}
x \quad (x>0)\\
0 \quad (x \le 0)
\end{cases}\\
&=max(0,x)
\end{aligned}
$$

![ReLU](images/2021-05-26-16-27-55.png)
