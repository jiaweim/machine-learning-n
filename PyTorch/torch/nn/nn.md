# torch.nn

- [torch.nn](#torchnn)
  - [简介](#简介)
  - [容器](#容器)
  - [卷积层](#卷积层)
  - [池化层](#池化层)
  - [填充层](#填充层)
  - [非线性激活（weighted sum, nonlinearity）](#非线性激活weighted-sum-nonlinearity)
  - [非线性激活（其它）](#非线性激活其它)
  - [归一化层](#归一化层)
  - [循环层](#循环层)
  - [Transformer](#transformer)
  - [线性层](#线性层)
  - [Dropout Layers](#dropout-layers)
  - [Loss Functions](#loss-functions)
    - [L1Loss](#l1loss)
    - [MSELoss](#mseloss)
    - [CrossEntropyLoss](#crossentropyloss)
  - [参考](#参考)

Last updated: 2023-01-09, 10:11
***

## 简介

图的基本构建模块。

## 容器

## 卷积层

## 池化层

## 填充层

## 非线性激活（weighted sum, nonlinearity）

## 非线性激活（其它）

## 归一化层

## 循环层

## Transformer

## 线性层

|Layer|说明|
|---|---|
|

## Dropout Layers

|Layer|说明|
|---|---|
nn.Dropout

During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.

nn.Dropout1d

Randomly zero out entire channels (a channel is a 1D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 1D tensor \text{input}[i, j]input[i,j]).

nn.Dropout2d

Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j]).

nn.Dropout3d

Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j]).

nn.AlphaDropout

Applies Alpha Dropout over the input.

nn.FeatureAlphaDropout

Randomly masks out entire channels (a channel is a feature map, e.g.

## Loss Functions

|损失函数|说明|
|---|---|
|[nn.L1Loss](#l1loss)|计算平均绝对误差（mean absolute error, MAE）|

nn.MSELoss

Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.

nn.CrossEntropyLoss

This criterion computes the cross entropy loss between input logits and target.

nn.CTCLoss

The Connectionist Temporal Classification loss.

nn.NLLLoss

The negative log likelihood loss.

nn.PoissonNLLLoss

Negative log likelihood loss with Poisson distribution of target.

nn.GaussianNLLLoss

Gaussian negative log likelihood loss.

nn.KLDivLoss

The Kullback-Leibler divergence loss.

nn.BCELoss

Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:

nn.BCEWithLogitsLoss

This loss combines a Sigmoid layer and the BCELoss in one single class.

nn.MarginRankingLoss

Creates a criterion that measures the loss given inputs x1x1, x2x2, two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor yy (containing 1 or -1).

nn.HingeEmbeddingLoss

Measures the loss given an input tensor xx and a labels tensor yy (containing 1 or -1).

nn.MultiLabelMarginLoss

Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input xx (a 2D mini-batch Tensor) and output yy (which is a 2D Tensor of target class indices).

nn.HuberLoss

Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.

nn.SmoothL1Loss

Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.

nn.SoftMarginLoss

Creates a criterion that optimizes a two-class classification logistic loss between input tensor xx and target tensor yy (containing 1 or -1).

nn.MultiLabelSoftMarginLoss

Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input xx and target yy of size (N, C)(N,C).

nn.CosineEmbeddingLoss

Creates a criterion that measures the loss given input tensors x_1x 
1
​
 , x_2x 
2
​
  and a Tensor label yy with values 1 or -1.

nn.MultiMarginLoss

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input xx (a 2D mini-batch Tensor) and output yy (which is a 1D tensor of target class indices, 0 \leq y \leq \text{x.size}(1)-10≤y≤x.size(1)−1):

nn.TripletMarginLoss

Creates a criterion that measures the triplet loss given an input tensors x1x1, x2x2, x3x3 and a margin with a value greater than 00.

nn.TripletMarginWithDistanceLoss

Creates a criterion that measures the triplet loss given input tensors aa, pp, and nn (representing anchor, positive, and negative examples, respectively), and a nonnegative, real-valued function ("distance function") used to compute the relationship between the anchor and positive example ("positive distance") and the anchor and negative example ("negative distance").

### L1Loss

```python
class torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
```

创建 input `x` 和 target `y` 每个元素之间的平均绝对误差（mean absolute error, MAE）。

非约简（即 `reduction='none'`）损失计算公式：

$$l(x,y)=L=\{l1,...,l_N\}^T, l_n=\lvert x_n-y_n\rvert$$

其中 $N$ 为 batch size。如果 `reduction` 不是 `'none'`（默认为 `'mean'`），则：

$$l(x,y)=\begin{cases}
  mean(L), & \text{reduction='mean';}\\
  sum(L), & \text{reduction='sum'}
\end{cases}$$

`x` 和 `y` 张量均包含 n 个元素。

如果设置 `reduction = 'sum'`，可以避免除以 $n$。

支持所有实数和复数输入。

**参数：**

- **size_average** (`bool`, optional) – **Deprecated** (see `reduction`)

`size_average` 表示是否计算 batch 所有损失的平均值。注意，对有些损失每个样本有多个损失值。`size_average`为 `False` 表示计算每个 minibatch 损失值加和。当 `reduce` 为 `False` 为 batch 的每个元素返回一个损失值，忽略 `size_average` 设置。默认 `True`。

- **reduce** (`bool`, optional) – **Deprecated** (see `reduction`)

默认根据 `size_average` 对每个 minibatch 的损失计算均值或加和。当 `reduce` 为 `False`，对 batch 的每个元素返回一个损失值，并忽略 `size_average` 设置。默认：`True`。

- **reduction** (`str`, optional)

指定应用于输出的约简操作：`'none'` | `'mean'` | `'sum'`.

`'none'`：不使用约简操作；`'mean'`： 计算均值；`'sum'`：计算加和。

默认：`'mean'`。

> **NOTE** `size_average` 和 `reduce` 即将被弃用，但是指定这两个参数中的任意一个都会覆盖 `reduction`。

**shape:**

- Input: `(*)`

`*` 表示任意维度数。

- Target: `(*)`

与输入相同 shape。

- Output: scalar

如果 `reduction` 为 `'none'`，则输出 shape 与输入相同。

示例：

```python
>>> from torch import nn
>>> import torch
>>> loss = nn.L1Loss()
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randn(3, 5)
>>> output = loss(input, target)
>>> output.backward()
```

### MSELoss

```python
class torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
```

计算 input `x` 和 target `y` 每个元素之间的均方误差（mean squared error, MSE）。

非约简（即 `reduction='none'`） 损失计算公式：

$$l(x,y)=L=\{l1,...,l_N\}^T, l_n=(x_n-y_n)^2$$

其中 $N$ 为 batch size。如果 `reduction` 不是 `'none'`（默认为 `'mean'`），则：

$$l(x,y)=\begin{cases}
  mean(L), & \text{reduction='mean';}\\
  sum(L), & \text{reduction='sum'}
\end{cases}$$

`x` 和 `y` 张量均包含 n 个元素。

如果设置 `reduction = 'sum'`，可以避免除以 $n$。

**参数：**

- **size_average** (`bool`, *optional*) – **Deprecated** (see `reduction`)

`size_average` 表示是否计算 batch 所有损失的平均值。注意，对有些损失每个样本有多个损失值。`size_average`为 `False` 表示计算每个 minibatch 损失值加和。`reduce=False` 表示不应用约简操作，为 batch 的每个元素返回一个损失值，此时忽略 `size_average` 设置。默认 `True`。

- **reduce** (`bool`, *optional*) – **Deprecated** (see `reduction`)

默认根据 `size_average` 对每个 minibatch 的损失计算均值或加和。`reduce=False` 表示对 batch 的每个元素返回一个损失值，并忽略 `size_average` 设置。默认：`True`。

- **reduction** (`str`, optional)

指定应用于输出的约简操作：`'none'` | `'mean'` | `'sum'`.

`'none'`：不使用约简操作；`'mean'`： 计算均值；`'sum'`：计算加和。

默认：`'mean'`。

> **NOTE** `size_average` 和 `reduce` 即将被弃用，但是指定这两个参数中的任意一个都会覆盖 `reduction`。

**shape:**

- Input: `(*)`

`*` 表示任意维度数。

- Target: `(*)`

与输入相同 shape。

示例：

```python
>>> loss = nn.MSELoss()
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randn(3, 5)
>>> output = loss(input, target)
>>> output.backward()
```

### CrossEntropyLoss

```python
class torch.nn.CrossEntropyLoss(
    weight=None, 
    size_average=None, 
    ignore_index=- 100, 
    reduce=None, 
    reduction='mean', 
    label_smoothing=0.0)
```

计算输入 logits 和 target 之间的交叉熵损失。



## 参考

- https://pytorch.org/docs/stable/nn.html
