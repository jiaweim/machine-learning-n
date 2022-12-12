# torch.Tensor

- [torch.Tensor](#torchtensor)
  - [简介](#简介)
  - [数据类型](#数据类型)
  - [初始化和基础操作](#初始化和基础操作)
  - [Tensor 类 API](#tensor-类-api)
  - [方法](#方法)
    - [detach](#detach)
    - [numpy](#numpy)
  - [参考](#参考)

***

## 简介

`torch.Tensor` 是包含单个数据类型元素的多维矩阵。

## 数据类型

Torch 定义了 10 种张量类型，包含 CPU 和 GPU 变体，如下：

|数据类型|dtype|CPU tensor|GPU tensor|
|---|---|---|---|
|32 位浮点数|`torch.float32` or `torch.float`|`torch.FloatTensor`|`torch.cuda.FloatTensor`|
|64 位浮点数|`torch.float64` or `torch.double`|`torch.DoubleTensor`|`torch.cuda.DoubleTensor`|
|16 位浮点数^[也称为 binary 16: 使用 1 个符号位，5 个指数位和 10 个有效位数，该类型牺牲范围换取精度]|`torch.float16` or `torch.half`|`torch.HalfTensor`|`torch.cuda.HalfTensor`|
|16 位浮点数^[也称为 Brain 浮点数：使用 1 个符号位，8 个指数位和 7 个有效位数，其指数位和 `float32` 相同，范围更大]|`torch.bfloat16`|`torch.BFloat16Tensor`|`torch.cuda.BFloat16Tensor`|
|32 位复数|`torch.complex32` or `torch.chalf`|
|64 位复数|`torch.complex64` or `torch.cfloat`|
|128 位复数|`torch.complex128` or `torch.cdouble`|
|8 位整数 (unsigned)|`torch.uint8`|`torch.ByteTensor`|`torch.cuda.ByteTensor`|
|8 位整数 (signed)|`torch.int8`|`torch.CharTensor`|`torch.cuda.CharTensor`|
|16 位整数 (signed)|`torch.int16` or `torch.short`|`torch.ShortTensor`|`torch.cuda.ShortTensor`|
|32 位整数 (signed)|`torch.int32` or `torch.int`|`torch.IntTensor`|`torch.cuda.IntTensor`|
|64 位整数 (signed)|`torch.int64` or `torch.long`|`torch.LongTensor`|`torch.cuda.LongTensor`|
|Boolean|`torch.bool`|`torch.BoolTensor`|`torch.cuda.BoolTensor`|
|quantized 8 位整数 (unsigned)|`torch.quint8`|`torch.ByteTensor`|/|
|quantized 8 位整数 (signed)|`torch.qint8`|`torch.CharTensor`|/|
|quantized 32 位整数 (signed)|`torch.qint32`|`torch.IntTensor`|/|
|quantized 4 位整数 (unsigned)^[quantized 4 位整数保存为 8 位 signed 整数，目前仅 EmbeddingBag 运算符支持]|`torch.quint4x2`|`torch.ByteTensor`|/|

`torch.Tensor` 的默认类型为 `torch.FloatTensor`。

## 初始化和基础操作

- 可以用 `torch.tensor()` 构造函数从 Python list 或 sequence 创建张量

```python
>>> torch.tensor([[1., -1.], [1., -1.]])
tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])
>>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
```

> **[!WARNING]** `torch.tensor()` 会复制底层数据 `data`。如果已有一个 Tensor 类型数据 `data`，只是想更改其 `requires_grad` 属性，可以用 `requires_grad_()` 或 `detach()` 来避免复制。如果已有一个 numpy 数组类型数据，可以用 `torch.as_tensor()` 来避免复制。

- 通过指定 `torch.dtype` 和 `torch.device` 可以创建特定类型的张量

```python
>>> torch.zeros([2, 4], dtype=torch.int32)
tensor([[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]], dtype=torch.int32)
>>> cuda0 = torch.device('cuda:0')
>>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')
```

- 张量的内容可以使用 Python 索引和切片进行访问和修改

```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> print(x[1][2])
tensor(6)
>>> x[0][1] = 8
>>> print(x)
tensor([[ 1,  8,  3],
        [ 4,  5,  6]])
```

- 对包含单个值的张量，可以使用 `torch.Tensor.item()` 返回 Python 类型值

```python
>>> x = torch.tensor([[1]])
>>> x
tensor([[ 1]])
>>> x.item()
1
>>> x = torch.tensor(2.5)
>>> x
tensor(2.5000)
>>> x.item()
2.5
```

- `torch.autograd` 会记录对 `requires_grad=True` 张量的操作，从而可以自动微分

```python
>>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
>>> out = x.pow(2).sum()
>>> out.backward()
>>> x.grad
tensor([[ 2.0000, -2.0000],
        [ 2.0000,  2.0000]])
```

每个张量都有一个关联的 `torch.Storage` 用来持有数据。

> **NOTE** 张量的原地操作方法以下划线结尾。例如，`torch.FloatTensor.abs_()` 原地计算绝对值，返回修改后的张量；而 `torch.FloatTensor.abs()` 的结果以新的张量返回。

> **NOTE** 修改已有张量的 `torch.device` 或 `torch.dtype`，可以调用 `to()` 方法。

> **WARNING** 当前 torch.Tensor 实现具有内存开销，创建大量小张量会导致高内存。对这种情况，建议使用一个大的张量结构。

## Tensor 类 API

```python
class torch.Tensor
```

创建张量的方法，主要有如下几种：

- 使用已有数据数据创建张量：`torch.tensor()`
- 创建指定 size 的张量：`torch.*`
- 创建与另一个张量相同 size 的张量：`torch.*_like`
- 创建于另一个张量相同 dtype 的张量：`tensor.new_*`

```python
Tensor.T
```

返回张量转置视图。

如果 `x` 的维度数为 `n`，则 `x.T` 等价于 `x.permute(n-1, n-2, ..., 0)`。

> **WARNING**：不推荐对维度不是 2 的张量使用 `Tensor.T()`，在未来版本会抛出错误。对批量矩阵的转置推荐使用 `mT`；反转矩阵维度推荐使用 `x.permute(*torch.arange(x.ndim - 1, -1, -1))`

```python
Tensor.H
```



## 方法

|方法|说明|
|---|---|
|[Tensor.detach](#detach) |返回一个从当前 graph 中分离出来的新的张量|

### detach

从当前图中分离出来，创建一个新的张量。

返回的张量不需要梯度。

该方法会影响正向模式的 AD 梯度，即结果也不会有正向模式的 AD 梯度。

### numpy

```python
Tensor.numpy(*, force=False) → numpy.ndarray
```

返回张量的 numpy 形式。

如果 `force` 为 `False`（默认），则要求 tensor 位于 CPU、不需要 grad、没有设置耦合位，dtype 和 layout  NumPy 支持才执行转换。返回的 ndarray 和 tensor 共享内存，因此张量和 ndarray 变化同步。

如果 `force` 为 `True`，则等价于 `t.detach().cpu().resolve_conj().resolve_neg().numpy()`。如果 tensor 不在 CPU，或者设置了共轭位或负位，则张量与 ndarray 不共享内存。将 `force` 设置为 `True` 是一种获得张量 ndarray 形式的简单方法。

即设置 `force=True` 返回的 ndarray 可以与张量不共享内存。

## 参考

- https://pytorch.org/docs/stable/tensors.html
