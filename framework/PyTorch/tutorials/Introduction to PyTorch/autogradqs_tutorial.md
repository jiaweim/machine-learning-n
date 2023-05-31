# 使用 torch.autograd 执行自动微分

- [使用 torch.autograd 执行自动微分](#使用-torchautograd-执行自动微分)
  - [简介](#简介)
  - [Tensor, Function 和计算图](#tensor-function-和计算图)
  - [计算梯度](#计算梯度)
  - [禁用梯度跟踪](#禁用梯度跟踪)
  - [计算图详解](#计算图详解)
  - [Tensor Gradient 和 Jacobian Product](#tensor-gradient-和-jacobian-product)
  - [参考](#参考)

Last updated: 2022-11-08, 13:57
****

## 简介

在训练神经网络时，**反向传播**是最常用的算法。该算法根据损失函数相对给定参数的**梯度**调整参数（模型权重）。

PyTorch 有一个内置的微分引擎 `torch.autograd`，它可以自动计算任意计算图的梯度。

例如，对最简单的单层神经网络，输入为 `x`，参数为 `w` 和 `b`，以及损失函数。在 PyTorch 能以如下方式定义：

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## Tensor, Function 和计算图

上面的代码定义了如下的**计算图**：

![](2022-11-08-11-15-38.png)

在该网络中，`w` 和 `b` 是待优化参数。要计算损失函数相对这些变量的梯度，设置这些张量的 `requires_grad` 属性即可。

> 可以在创建张量时设置 `requires_grad`，也可以创建张量后调用 x`.requires_grad_(True)`

根据张量构造计算图的函数实际是 `Function` 类的对象。该对象知道如何在 *forward* 方向计算函数，也知道如何在 *backward propagation* 时计算导数。张量的 `grad_fn` 属性存储反向传播函数的引用。更多信息可参考 [Function 文档](https://pytorch.org/docs/stable/autograd.html#function)。

```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

```txt
Gradient function for z = <AddBackward0 object at 0x00000136137AA220>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x0000013613A4A220>
```

## 计算梯度

为了优化神经网络的权重参数，需要计算损失函数对参数的导数，即在指定 x 和 y 下计算 $\frac{\partial loss}{\partial w}$ 和 $\frac{\partial loss}{\partial b}$。调用 `loss.backward()` 即可计算这些导数，然后从 `w.grad` 和 `b.grad` 查看梯度：

```python
loss.backward()
print(w.grad)
print(b.grad)
```

```txt
tensor([[0.1704, 0.3327, 0.1361],
        [0.1704, 0.3327, 0.1361],
        [0.1704, 0.3327, 0.1361],
        [0.1704, 0.3327, 0.1361],
        [0.1704, 0.3327, 0.1361]])
tensor([0.1704, 0.3327, 0.1361])
```

> **NOTE**
> - 我们只能获得计算图的叶节点的 `grad` 属性，它们的 `requires_grad` 为 `True`。计算图中其它节点的梯度不可用。
> - 出于性能原因，只能在给定的图上使用 `backward` 执行一次梯度计算。要在同一个计算图上执行多次 `backward`，在调用 `backward` 要传入 `retain_graph=True`。

## 禁用梯度跟踪

默认跟踪所有 `requires_grad=True` 的张量的计算历史，并支持梯度计算。然而有时候不需要计算梯度，如使用训练好的模型预测新的输入数据，此时只需要前向计算。`torch.no_grad()` 包围的代码块禁用跟踪计算：

```python
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
```

```txt
True
False
```

另一种禁用梯度的方式是调用张量的 `detach()` 方法：

```python
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)
```

```txt
False
```

禁用梯度跟踪的主要原因有：

- 将神经网络中的一些参数标记为**冻结参数**。对预训练网络进行微调中常用。
- 只做前向传播时禁用梯度跟踪可以加速计算。

## 计算图详解

从概念上讲，autograd 记录由 [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) 对象组成的有向无环图（DAG）中所有数据和操作。在 DAG 中，叶节点是输入张量，根节点是输出张量。通过 DAG 中从根节点到叶节点的路径，可以使用链式法则计算梯度。

在前向传播时，autograd 同时做两件事：

- 执行请求的操作，计算得到张量
- 维护 DAG 中操作的梯度函数

在 DAG 根节点上调用 `.backward()` 开始反向传播。autograd 执行：

- 对每个 `.grad_fn` 计算梯度
- 将梯度更新到各个张量的 `.grad` 属性
- 使用链式法则，传播梯度到叶节点张量

> **NOTE**
> **PyTorch 中的 DAG 是动态的**。需要注意的是，graph 是从头创建的，每次调用 `.backward()` 后，autograd 开始填充新的 graph。这是允许在模型中使用控制流语句的原因，如果需要，可以在每次迭代更改 shape, size 以及操作。

## Tensor Gradient 和 Jacobian Product

大多时候，我们有一个损失函数计算得到的标量损失值，然后需要计算损失函数关于一些参数的梯度。然而，也有输出是张量的情况。PyTorch 允许计算 Jacobian 乘积，而不是实际的梯度。

对向量函数 $\vec{y}=f(\vec{x})$，其中 $\vec{x}=\langle x_1,\dots,x_n\rangle$, $\vec{y}=\langle y_1,\dots,y_m\rangle$，$\vec{y}$ 相对 $\vec{x}$ 的梯度由 Jacobian 矩阵（雅可比矩阵）给出：

$$J=\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix}$$

不是计算 Jacobian 矩阵本身，PyTorch 可以计算输入向量 $v=(v_1 \cdots v_m)$ 的 Jacobian 乘积 $v^T \cdot J$。以 $v$ 为参数调用 backward 计算 Jacobian 乘积。$v$ 的 size 应该与原始张量相同：

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

```txt
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
```

注意，当用相同参数第二次调用 backward 时，梯度的值不同。这是因为在反向传播时，PyTorch 会**累计梯度**，即计算的梯度值被添加到计算图的所有叶节点的 `grad` 属性。如果要计算正确的梯度，需要提前把 `grad` 属性归零。在实际训练中，优化器会自动执行该操作。

> **NOTE**
> 之前调用 `backward()` 函数不带参数，这等价于调用 `backward(torch.tensor(1.0))`，这对损失函数的输出为标量值时很有用。

## 参考

- https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
