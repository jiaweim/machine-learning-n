# 张量（Tensor）

2026-09-11: 基于 2.14.0 API 修改
@since 2023-02-06⭐
@author Jiawei Mao
***

## 简介

张量是一种特殊的数据结构，与数组和矩阵类似。PyTorch 使用张量来编码模型的输入、输出以及模型的参数。

张量与 NumPy 的 `ndarray` 类似，不同之处在于，张量可以在 GPU 等硬件加速器上运行，并对自动微分进行了优化。实际上，张量和 NumPy 数组通常可以共享内存，从而避免复制数据（参考 [与 NumPy 互转](#与-numpy-互转)）。如果熟悉 ndarray，那么掌握 Tensor API 没有难度。

```python
import torch
import numpy as np
```

## 创建张量

创建张量的方式有多种。

### 从数据创建

直接从数据创建张量，自动推断数据类型：

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

### 从 NumPy 数组创建

可以直接从 NumPy 数组创建张量，反之亦然

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

### 从其它张量创建

新张量保留参数张量的属性（shape, dtype），除非显式覆盖：

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

### 使用随机数或常数初始化

`shape` 是表示张量维度的 tuple。下面的函数使用 `shape` 参数设置输出张量的维度：

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

张量属性包括：

- 形状：`shape`
- 数据类型：`dtype`
- 存储设备：`device`

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

张量操作有 1200 多个，包括算术运算、线性代数、矩阵操作、采样等，具体参考 [详细列表](https://pytorch.org/docs/stable/torch.html)。

这些操作都可以在 CPU 以及各类硬件加速器（如 CUDA、MPS、MITA或 XPU）上运行（通常比在 CPU 上快）。

默认在 CPU 上创建张量，可以使用 `.to` 方法将张量移动到 GPU。注意，跨设备复制大型张量比较占用时间和内存。

```python
# 如果当前有可用的加速器，将张量移动到该加速器上
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```

下面演示张量操作。

### 标准 numpy-like 索引和切片

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}") # : 表示选取所有 rows
# ... 表示前面所有维度，对二维张量与 : 等价
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

### 合并张量

可以使用 `torch.cat` 将多个张量沿指定维度拼接起来。

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

### 算术运算

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

### 单元素张量

可以使用 `item()` 将单元素张量转换为 Python 值：

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```txt
12.0 <class 'float'>
```

### 原地操作（in-place）

将运算结果保存到操作数（operand）自身的操作称为**原地操作**（in-place）。它们由 `_` 后缀标识。例如 `x.copy_(y)`, `x.t_()` 会为修改 `x`。

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

> [!NOTE]
>
> 原地操作虽然会节省一些内存，但在计算梯度时可能引发问题，因为会丢失计算历史。因此不推荐使用原地操作。

## 与 NumPy 互转

**CPU** 上的张量与 NumPy 数组可以共享底层内存，修改一个会同步更改另一个。

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
