# 张量

- [张量](#张量)
  - [简介](#简介)
  - [创建张量](#创建张量)
  - [张量属性](#张量属性)
  - [张量操作](#张量操作)
  - [与 NumPy 互转](#与-numpy-互转)
    - [Tensor 到 NumPy](#tensor-到-numpy)
    - [NumPy 到 Tensor](#numpy-到-tensor)
  - [参考](#参考)

Last updated: 2022-11-07, 15:48
****

## 简介

张量是一种特殊的数据结构，与数组和矩阵特别像。在 PyTorch，使用张量来编码模型的输入和输出，以及模型的参数。

张量与 NumPy 的 ndarray 类似，不同之处在于张量可以在 GPU 等其它硬件加速器上运行，并对自动微分进行了优化。实际上，张量和 NumPy 数组通常可以共享内存，从而避免了复制数据（参考 [与 NumPy 互转](#与-numpy-互转)）。如果熟悉 ndarray，那么掌握 Tensor API 没有难度。

```python
import torch
import numpy as np
```

## 创建张量

可以用以下方式创建张量。

**直接从数据创建**

可以直接从数据创建张量，自动推断数据类型：

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

**从 NumPy 数组创建**

可以直接从 NumPy 数组创建张量，反之亦然

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**从其它张量创建**

新的张量保留参数张量的属性（shape, dtype），除非显式覆盖：

```python
x_ones = torch.ones_like(x_data)  # 和 x_data 属性相同
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")
```

```txt
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.4130, 0.6221],
        [0.6819, 0.9206]]) 
```

**随机数或常数**

`shape` 是张量维度的 tuple。在下面的函数中，它决定输出张量的维度：

```python
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")
```

```txt
Random Tensor: 
 tensor([[0.4211, 0.3871, 0.3292],
        [0.7483, 0.4832, 0.8292]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]]) 
```

## 张量属性

张量属性描述了其形状、数据类型和存储设备。

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

```txt
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

## 张量操作

张量操作有 100 多个，包括算术、线性代数、矩阵操作、采样等，具体参考 [详细列表](https://pytorch.org/docs/stable/torch.html)。

这些操作都可以在 GPU 上运行（通常比在 CPU 上快）。

默认在 CPU 上创建张量，可以使用 `.to` 方法将张量移动到 GPU（确定 GPU 可用后）。注意，跨设备复制大型张量比较占用时间和内存。

```python
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

张量操作演示：

**标准 numpy-like 索引和切片**

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)
```

```txt
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**连接张量**

可以使用 `torch.cat` 将一系列张量沿指定维度连接起来。

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```txt
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**算术运算**

```python
# 计算矩阵乘，y1, y2, y3 的值相同
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# 计算逐元素乘，z1, z2, z3 值相同
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

```txt
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**单元素张量**

可以使用 `item()` 将单元素张量转换为 Python 值：

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```txt
12.0 <class 'float'>
```

**就地操作**

将值保存到操作数（operand）的操作称为就地操作。它们由 `_` 后缀标识。例如 `x.copy_(y)`, `x.t_()` 会为修改 `x`。

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

```txt
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

> **Note:** 就地操作会节省一些内存，但在计算导数时会出现问题，因为会丢失历史记录。因此不推荐使用就地操作。

## 与 NumPy 互转

CPU 上的张量与 NumPy 数组可以共享底层内存，修改一个会同步更改另一个。

### Tensor 到 NumPy

`.numpy()` 转换为 NumPy 数组。
 
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```txt
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

修改张量 NumPy 数组也随之更改:

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```txt
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy 到 Tensor

`torch.from_numpy` 从 NumPy 数组生成张量。

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

修改 NumPy 数组张量也随之改变：

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```txt
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
