# torch.Tensor

- [torch.Tensor](#torchtensor)
  - [简介](#简介)
  - [数据类型](#数据类型)
  - [初始化和基础操作](#初始化和基础操作)
  - [Tensor 类 API](#tensor-类-api)
    - [permute](#permute)
    - [repeat](#repeat)
  - [操作](#操作)
    - [detach](#detach)
    - [numpy](#numpy)
    - [scatter\_](#scatter_)
    - [Tensor.to](#tensorto)
    - [view](#view)
    - [Tensor.byte](#tensorbyte)
    - [Tensor.cpu](#tensorcpu)
    - [Tensor.cuda](#tensorcuda)
    - [Tensor.type](#tensortype)
  - [参考](#参考)

Last updated: 2022-12-13, 17:34
****

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

### permute

```python
Tensor.permute(*dims) → Tensor
```

参考 `torch.permute()`。

### repeat

```python
Tensor.repeat(*sizes) → Tensor
```

沿指定维度重复张量。

和 `expand()` 不同，该函数会复制张量数据。

**参数：**

- **sizes** (`torch.Size` or `int`...)

指定在各个维度重复的次数。

```python
>>> x = torch.tensor([1, 2, 3])
>>> x.repeat(4, 2)
tensor([[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]])
>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])
```

## 操作

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

### scatter_

```python
Tensor.scatter_(dim, index, src, reduce=None) → Tensor
```

按照索引张量 `index` 将 `src` 张量的值写入 `self`。对 `src` 的每个值，其输出索引由 `src` 的索引（`dimension != dim`）和 `index` 中的对应值指定（`dimension = dim`）。

对 3D 张量，`self` 更新方式：

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

对 2D 张量，`self` 的更新方式：

```python
self[index[i][j]][j] = src[i][j] # if dim == 0
```

该操作和 `gather()` 操作相反。

需要注意：

- `self`, `index` 和 `src` 的维数必须相同；
- 对所有维度 `d`，要求 `index.size(d) <= src.size(d)`；
- 对所有 `d != dim` 的维度，要求 `index.size(d) <= self.size(d)`；
- `index` 和 `src` 不广播

> **WARNING** 当 indices 不

参数：

- **dim** (`int`)：索引维度；
- **index** (`LongTensor`)：需分配元素的索引，可以为空或与 `src` 相同维度，为空时返回 `self` 不变；
- **src** (`Tensor` 或 `float`)：待分配的元素
- **reduce** (`str`, optional)：缩减操作，如 add 或 multiply。

例如：

```python
>>> src = torch.arange(1, 11).reshape((2, 5))
>>> src # shape (2, 5)
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]]) # (1, 4)
# output[index[i][j]][j] = src[i][j]
# src[0][0]=self[index[0][0]][0]=self[0][0]=1
# src[0][1]=self[index[0][1]][1]=self[1][1]=2
# src[0][2]=self[index[0][2]][2]=self[2][2]=3
# src[0][3]=self[index[0][3]][3]=self[0][3]=4
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src) # self (3, 5)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])

>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])

>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='multiply')
tensor([[2.0000, 2.0000, 2.4600, 2.0000],
        [2.0000, 2.0000, 2.0000, 2.4600]])
>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='add')
tensor([[2.0000, 2.0000, 3.2300, 2.0000],
        [2.0000, 2.0000, 2.0000, 3.2300]])
```

### Tensor.to

Last updated: 2023-01-30, 16:48

```python
Tensor.to(*args, **kwargs) → Tensor
```

执行张量 `dtype` and/or `device` 转换。`torch.dtype` 和 `torch.device` 
 
```python
>>> long_tensor = torch.tensor([[0, 0, 1], [1, 1, 1], [0, 0, 0]])
>>> long_tensor.type()
'torch.LongTensor'
>>> float_tensor = long_tensor.to(dtype=torch.float32)
>>> float_tensor.type()
'torch.FloatTensor'
```

### view

```python
Tensor.view(*shape) → Tensor
```

返回一个与 `self` 张量具有相同数据是 shape 不同的张量。

要张量视图，新的视图尺寸与原始张量的尺寸和步长必须兼容，即新的视图维度要么是原始维度的子空间，要么

Tensor.new_tensor

Returns a new Tensor with data as the tensor data.

Tensor.new_full

Returns a Tensor of size size filled with fill_value.

Tensor.new_empty

Returns a Tensor of size size filled with uninitialized data.

Tensor.new_ones

Returns a Tensor of size size filled with 1.

Tensor.new_zeros

Returns a Tensor of size size filled with 0.

Tensor.is_cuda

Is True if the Tensor is stored on the GPU, False otherwise.

Tensor.is_quantized

Is True if the Tensor is quantized, False otherwise.

Tensor.is_meta

Is True if the Tensor is a meta tensor, False otherwise.

Tensor.device

Is the torch.device where this Tensor is.

Tensor.grad

This attribute is None by default and becomes a Tensor the first time a call to backward() computes gradients for self.

Tensor.ndim

Alias for dim()

Tensor.real

Returns a new tensor containing real values of the self tensor for a complex-valued input tensor.

Tensor.imag

Returns a new tensor containing imaginary values of the self tensor.

Tensor.abs

See torch.abs()

Tensor.abs_

In-place version of abs()

Tensor.absolute

Alias for abs()

Tensor.absolute_

In-place version of absolute() Alias for abs_()

Tensor.acos

See torch.acos()

Tensor.acos_

In-place version of acos()

Tensor.arccos

See torch.arccos()

Tensor.arccos_

In-place version of arccos()

Tensor.add

Add a scalar or tensor to self tensor.

Tensor.add_

In-place version of add()

Tensor.addbmm

See torch.addbmm()

Tensor.addbmm_

In-place version of addbmm()

Tensor.addcdiv

See torch.addcdiv()

Tensor.addcdiv_

In-place version of addcdiv()

Tensor.addcmul

See torch.addcmul()

Tensor.addcmul_

In-place version of addcmul()

Tensor.addmm

See torch.addmm()

Tensor.addmm_

In-place version of addmm()

Tensor.sspaddmm

See torch.sspaddmm()

Tensor.addmv

See torch.addmv()

Tensor.addmv_

In-place version of addmv()

Tensor.addr

See torch.addr()

Tensor.addr_

In-place version of addr()

Tensor.adjoint

Alias for adjoint()

Tensor.allclose

See torch.allclose()

Tensor.amax

See torch.amax()

Tensor.amin

See torch.amin()

Tensor.aminmax

See torch.aminmax()

Tensor.angle

See torch.angle()

Tensor.apply_

Applies the function callable to each element in the tensor, replacing each element with the value returned by callable.

Tensor.argmax

See torch.argmax()

Tensor.argmin

See torch.argmin()

Tensor.argsort

See torch.argsort()

Tensor.argwhere

See torch.argwhere()

Tensor.asin

See torch.asin()

Tensor.asin_

In-place version of asin()

Tensor.arcsin

See torch.arcsin()

Tensor.arcsin_

In-place version of arcsin()

Tensor.as_strided

See torch.as_strided()

Tensor.atan

See torch.atan()

Tensor.atan_

In-place version of atan()

Tensor.arctan

See torch.arctan()

Tensor.arctan_

In-place version of arctan()

Tensor.atan2

See torch.atan2()

Tensor.atan2_

In-place version of atan2()

Tensor.arctan2

See torch.arctan2()

Tensor.arctan2_

atan2_(other) -> Tensor

Tensor.all

See torch.all()

Tensor.any

See torch.any()

Tensor.backward

Computes the gradient of current tensor w.r.t.

Tensor.baddbmm

See torch.baddbmm()

Tensor.baddbmm_

In-place version of baddbmm()

Tensor.bernoulli

Returns a result tensor where each 
result[i]
result[i] is independently sampled from 
Bernoulli
(
self[i]
)
Bernoulli(self[i]).

Tensor.bernoulli_

Fills each location of self with an independent sample from 
Bernoulli
(
p
)
Bernoulli(p).

Tensor.bfloat16

self.bfloat16() is equivalent to self.to(torch.bfloat16).

Tensor.bincount

See torch.bincount()

Tensor.bitwise_not

See torch.bitwise_not()

Tensor.bitwise_not_

In-place version of bitwise_not()

Tensor.bitwise_and

See torch.bitwise_and()

Tensor.bitwise_and_

In-place version of bitwise_and()

Tensor.bitwise_or

See torch.bitwise_or()

Tensor.bitwise_or_

In-place version of bitwise_or()

Tensor.bitwise_xor

See torch.bitwise_xor()

Tensor.bitwise_xor_

In-place version of bitwise_xor()

Tensor.bitwise_left_shift

See torch.bitwise_left_shift()

Tensor.bitwise_left_shift_

In-place version of bitwise_left_shift()

Tensor.bitwise_right_shift

See torch.bitwise_right_shift()

Tensor.bitwise_right_shift_

In-place version of bitwise_right_shift()

Tensor.bmm

See torch.bmm()

Tensor.bool

self.bool() is equivalent to self.to(torch.bool).

### Tensor.byte

Last updated: 2023-01-30, 17:01

```python
Tensor.byte(memory_format=torch.preserve_format) → Tensor
```

`self.byte()` 等价于 `self.to(torch.uint8)`。

**参数：**

- **memory_format** (`torch.memory_format`, optional)

张量的内存格式。

Tensor.broadcast_to

See torch.broadcast_to().

Tensor.cauchy_

Fills the tensor with numbers drawn from the Cauchy distribution:

Tensor.ceil

See torch.ceil()

Tensor.ceil_

In-place version of ceil()

Tensor.char

self.char() is equivalent to self.to(torch.int8).

Tensor.cholesky

See torch.cholesky()

Tensor.cholesky_inverse

See torch.cholesky_inverse()

Tensor.cholesky_solve

See torch.cholesky_solve()

Tensor.chunk

See torch.chunk()

Tensor.clamp

See torch.clamp()

Tensor.clamp_

In-place version of clamp()

Tensor.clip

Alias for clamp().

Tensor.clip_

Alias for clamp_().

Tensor.clone

See torch.clone()

Tensor.contiguous

Returns a contiguous in memory tensor containing the same data as self tensor.

Tensor.copy_

Copies the elements from src into self tensor and returns self.

Tensor.conj

See torch.conj()

Tensor.conj_physical

See torch.conj_physical()

Tensor.conj_physical_

In-place version of conj_physical()

Tensor.resolve_conj

See torch.resolve_conj()

Tensor.resolve_neg

See torch.resolve_neg()

Tensor.copysign

See torch.copysign()

Tensor.copysign_

In-place version of copysign()

Tensor.cos

See torch.cos()

Tensor.cos_

In-place version of cos()

Tensor.cosh

See torch.cosh()

Tensor.cosh_

In-place version of cosh()

Tensor.corrcoef

See torch.corrcoef()

Tensor.count_nonzero

See torch.count_nonzero()

Tensor.cov

See torch.cov()

Tensor.acosh

See torch.acosh()

Tensor.acosh_

In-place version of acosh()

Tensor.arccosh

acosh() -> Tensor

Tensor.arccosh_

acosh_() -> Tensor

### Tensor.cpu

Last updated: 2023-01-30, 16:13

```python
Tensor.cpu(memory_format=torch.preserve_format) → Tensor
```

返回张量的 CPU 副本。

如果张量已在 CPU 内存且处于正确设备，则不复制，直接返回已有对象。

**参数：**

- **memory_format** (`torch.memory_format`, optional)

返回张量的内存格式，默认 `torch.preserve_format`。


Tensor.cross

See torch.cross()

### Tensor.cuda

Last updated: 2023-01-30, 16:10

```python
Tensor.cuda(device=None, 
    non_blocking=False, 
    memory_format=torch.preserve_format) → Tensor
```

返回张量的 CUDA 副本。

如果该张量已在 CUDA 内存且设备没错，则不复制，直接返回原对象。

**参数：**

- **device=None** (`torch.device`)

目标 GPU 设备，默认为当前 CUDA 设备。

- **non_blocking=False** (`bool`)

设置 `True` 且原张量位于 pinned 内存，则相对主机复制是异步执行的。否则该参数无效。

- **memory_format** (`torch.memory_format`, optional)

返回张量的内存格式。默认 `torch.preserve_format`。

Tensor.logcumsumexp

See torch.logcumsumexp()

Tensor.cummax

See torch.cummax()

Tensor.cummin

See torch.cummin()

Tensor.cumprod

See torch.cumprod()

Tensor.cumprod_

In-place version of cumprod()

Tensor.cumsum

See torch.cumsum()

Tensor.cumsum_

In-place version of cumsum()

Tensor.chalf

self.chalf() is equivalent to self.to(torch.complex32).

Tensor.cfloat

self.cfloat() is equivalent to self.to(torch.complex64).

Tensor.cdouble

self.cdouble() is equivalent to self.to(torch.complex128).

Tensor.data_ptr

Returns the address of the first element of self tensor.

Tensor.deg2rad

See torch.deg2rad()

Tensor.dequantize

Given a quantized Tensor, dequantize it and return the dequantized float Tensor.

Tensor.det

See torch.det()

Tensor.dense_dim

Return the number of dense dimensions in a sparse tensor self.

Tensor.detach

Returns a new Tensor, detached from the current graph.

Tensor.detach_

Detaches the Tensor from the graph that created it, making it a leaf.

Tensor.diag

See torch.diag()

Tensor.diag_embed

See torch.diag_embed()

Tensor.diagflat

See torch.diagflat()

Tensor.diagonal

See torch.diagonal()

Tensor.diagonal_scatter

See torch.diagonal_scatter()

Tensor.fill_diagonal_

Fill the main diagonal of a tensor that has at least 2-dimensions.

Tensor.fmax

See torch.fmax()

Tensor.fmin

See torch.fmin()

Tensor.diff

See torch.diff()

Tensor.digamma

See torch.digamma()

Tensor.digamma_

In-place version of digamma()

Tensor.dim

Returns the number of dimensions of self tensor.

Tensor.dist

See torch.dist()

Tensor.div

See torch.div()

Tensor.div_

In-place version of div()

Tensor.divide

See torch.divide()

Tensor.divide_

In-place version of divide()

Tensor.dot

See torch.dot()

Tensor.double

self.double() is equivalent to self.to(torch.float64).

Tensor.dsplit

See torch.dsplit()

Tensor.element_size

Returns the size in bytes of an individual element.

Tensor.eq

See torch.eq()

Tensor.eq_

In-place version of eq()

Tensor.equal

See torch.equal()

Tensor.erf

See torch.erf()

Tensor.erf_

In-place version of erf()

Tensor.erfc

See torch.erfc()

Tensor.erfc_

In-place version of erfc()

Tensor.erfinv

See torch.erfinv()

Tensor.erfinv_

In-place version of erfinv()

Tensor.exp

See torch.exp()

Tensor.exp_

In-place version of exp()

Tensor.expm1

See torch.expm1()

Tensor.expm1_

In-place version of expm1()

Tensor.expand

Returns a new view of the self tensor with singleton dimensions expanded to a larger size.

Tensor.expand_as

Expand this tensor to the same size as other.

Tensor.exponential_

Fills self tensor with elements drawn from the exponential distribution:

Tensor.fix

See torch.fix().

Tensor.fix_

In-place version of fix()

Tensor.fill_

Fills self tensor with the specified value.

Tensor.flatten

See torch.flatten()

Tensor.flip

See torch.flip()

Tensor.fliplr

See torch.fliplr()

Tensor.flipud

See torch.flipud()

Tensor.float

self.float() is equivalent to self.to(torch.float32).

Tensor.float_power

See torch.float_power()

Tensor.float_power_

In-place version of float_power()

Tensor.floor

See torch.floor()

Tensor.floor_

In-place version of floor()

Tensor.floor_divide

See torch.floor_divide()

Tensor.floor_divide_

In-place version of floor_divide()

Tensor.fmod

See torch.fmod()

Tensor.fmod_

In-place version of fmod()

Tensor.frac

See torch.frac()

Tensor.frac_

In-place version of frac()

Tensor.frexp

See torch.frexp()

Tensor.gather

See torch.gather()

Tensor.gcd

See torch.gcd()

Tensor.gcd_

In-place version of gcd()

Tensor.ge

See torch.ge().

Tensor.ge_

In-place version of ge().

Tensor.greater_equal

See torch.greater_equal().

Tensor.greater_equal_

In-place version of greater_equal().

Tensor.geometric_

Fills self tensor with elements drawn from the geometric distribution:

Tensor.geqrf

See torch.geqrf()

Tensor.ger

See torch.ger()

Tensor.get_device

For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.

Tensor.gt

See torch.gt().

Tensor.gt_

In-place version of gt().

Tensor.greater

See torch.greater().

Tensor.greater_

In-place version of greater().

Tensor.half

self.half() is equivalent to self.to(torch.float16).

Tensor.hardshrink

See torch.nn.functional.hardshrink()

Tensor.heaviside

See torch.heaviside()

Tensor.histc

See torch.histc()

Tensor.histogram

See torch.histogram()

Tensor.hsplit

See torch.hsplit()

Tensor.hypot

See torch.hypot()

Tensor.hypot_

In-place version of hypot()

Tensor.i0

See torch.i0()

Tensor.i0_

In-place version of i0()

Tensor.igamma

See torch.igamma()

Tensor.igamma_

In-place version of igamma()

Tensor.igammac

See torch.igammac()

Tensor.igammac_

In-place version of igammac()

Tensor.index_add_

Accumulate the elements of alpha times source into the self tensor by adding to the indices in the order given in index.

Tensor.index_add

Out-of-place version of torch.Tensor.index_add_().

Tensor.index_copy_

Copies the elements of tensor into the self tensor by selecting the indices in the order given in index.

Tensor.index_copy

Out-of-place version of torch.Tensor.index_copy_().

Tensor.index_fill_

Fills the elements of the self tensor with value value by selecting the indices in the order given in index.

Tensor.index_fill

Out-of-place version of torch.Tensor.index_fill_().

Tensor.index_put_

Puts values from the tensor values into the tensor self using the indices specified in indices (which is a tuple of Tensors).

Tensor.index_put

Out-place version of index_put_().

Tensor.index_reduce_

Accumulate the elements of source into the self tensor by accumulating to the indices in the order given in index using the reduction given by the reduce argument.

Tensor.index_reduce

Tensor.index_select

See torch.index_select()

Tensor.indices

Return the indices tensor of a sparse COO tensor.

Tensor.inner

See torch.inner().

Tensor.int

self.int() is equivalent to self.to(torch.int32).

Tensor.int_repr

Given a quantized Tensor, self.int_repr() returns a CPU Tensor with uint8_t as data type that stores the underlying uint8_t values of the given Tensor.

Tensor.inverse

See torch.inverse()

Tensor.isclose

See torch.isclose()

Tensor.isfinite

See torch.isfinite()

Tensor.isinf

See torch.isinf()

Tensor.isposinf

See torch.isposinf()

Tensor.isneginf

See torch.isneginf()

Tensor.isnan

See torch.isnan()

Tensor.is_contiguous

Returns True if self tensor is contiguous in memory in the order specified by memory format.

Tensor.is_complex

Returns True if the data type of self is a complex data type.

Tensor.is_conj

Returns True if the conjugate bit of self is set to true.

Tensor.is_floating_point

Returns True if the data type of self is a floating point data type.

Tensor.is_inference

See torch.is_inference()

Tensor.is_leaf

All Tensors that have requires_grad which is False will be leaf Tensors by convention.

Tensor.is_pinned

Returns true if this tensor resides in pinned memory.

Tensor.is_set_to

Returns True if both tensors are pointing to the exact same memory (same storage, offset, size and stride).

Tensor.is_shared

Checks if tensor is in shared memory.

Tensor.is_signed

Returns True if the data type of self is a signed data type.

Tensor.is_sparse

Is True if the Tensor uses sparse storage layout, False otherwise.

Tensor.istft

See torch.istft()

Tensor.isreal

See torch.isreal()

Tensor.item

Returns the value of this tensor as a standard Python number.

Tensor.kthvalue

See torch.kthvalue()

Tensor.lcm

See torch.lcm()

Tensor.lcm_

In-place version of lcm()

Tensor.ldexp

See torch.ldexp()

Tensor.ldexp_

In-place version of ldexp()

Tensor.le

See torch.le().

Tensor.le_

In-place version of le().

Tensor.less_equal

See torch.less_equal().

Tensor.less_equal_

In-place version of less_equal().

Tensor.lerp

See torch.lerp()

Tensor.lerp_

In-place version of lerp()

Tensor.lgamma

See torch.lgamma()

Tensor.lgamma_

In-place version of lgamma()

Tensor.log

See torch.log()

Tensor.log_

In-place version of log()

Tensor.logdet

See torch.logdet()

Tensor.log10

See torch.log10()

Tensor.log10_

In-place version of log10()

Tensor.log1p

See torch.log1p()

Tensor.log1p_

In-place version of log1p()

Tensor.log2

See torch.log2()

Tensor.log2_

In-place version of log2()

Tensor.log_normal_

Fills self tensor with numbers samples from the log-normal distribution parameterized by the given mean 
�
μ and standard deviation 
�
σ.

Tensor.logaddexp

See torch.logaddexp()

Tensor.logaddexp2

See torch.logaddexp2()

Tensor.logsumexp

See torch.logsumexp()

Tensor.logical_and

See torch.logical_and()

Tensor.logical_and_

In-place version of logical_and()

Tensor.logical_not

See torch.logical_not()

Tensor.logical_not_

In-place version of logical_not()

Tensor.logical_or

See torch.logical_or()

Tensor.logical_or_

In-place version of logical_or()

Tensor.logical_xor

See torch.logical_xor()

Tensor.logical_xor_

In-place version of logical_xor()

Tensor.logit

See torch.logit()

Tensor.logit_

In-place version of logit()

Tensor.long

self.long() is equivalent to self.to(torch.int64).

Tensor.lt

See torch.lt().

Tensor.lt_

In-place version of lt().

Tensor.less

lt(other) -> Tensor

Tensor.less_

In-place version of less().

Tensor.lu

See torch.lu()

Tensor.lu_solve

See torch.lu_solve()

Tensor.as_subclass

Makes a cls instance with the same data pointer as self.

Tensor.map_

Applies callable for each element in self tensor and the given tensor and stores the results in self tensor.

Tensor.masked_scatter_

Copies elements from source into self tensor at positions where the mask is True.

Tensor.masked_scatter

Out-of-place version of torch.Tensor.masked_scatter_()

Tensor.masked_fill_

Fills elements of self tensor with value where mask is True.

Tensor.masked_fill

Out-of-place version of torch.Tensor.masked_fill_()

Tensor.masked_select

See torch.masked_select()

Tensor.matmul

See torch.matmul()

Tensor.matrix_power

NOTE

matrix_power() is deprecated, use torch.linalg.matrix_power() instead.

Tensor.matrix_exp

See torch.matrix_exp()

Tensor.max

See torch.max()

Tensor.maximum

See torch.maximum()

Tensor.mean

See torch.mean()

Tensor.nanmean

See torch.nanmean()

Tensor.median

See torch.median()

Tensor.nanmedian

See torch.nanmedian()

Tensor.min

See torch.min()

Tensor.minimum

See torch.minimum()

Tensor.mm

See torch.mm()

Tensor.smm

See torch.smm()

Tensor.mode

See torch.mode()

Tensor.movedim

See torch.movedim()

Tensor.moveaxis

See torch.moveaxis()

Tensor.msort

See torch.msort()

Tensor.mul

See torch.mul().

Tensor.mul_

In-place version of mul().

Tensor.multiply

See torch.multiply().

Tensor.multiply_

In-place version of multiply().

Tensor.multinomial

See torch.multinomial()

Tensor.mv

See torch.mv()

Tensor.mvlgamma

See torch.mvlgamma()

Tensor.mvlgamma_

In-place version of mvlgamma()

Tensor.nansum

See torch.nansum()

Tensor.narrow

See torch.narrow()

Tensor.narrow_copy

See torch.narrow_copy().

Tensor.ndimension

Alias for dim()

Tensor.nan_to_num

See torch.nan_to_num().

Tensor.nan_to_num_

In-place version of nan_to_num().

Tensor.ne

See torch.ne().

Tensor.ne_

In-place version of ne().

Tensor.not_equal

See torch.not_equal().

Tensor.not_equal_

In-place version of not_equal().

Tensor.neg

See torch.neg()

Tensor.neg_

In-place version of neg()

Tensor.negative

See torch.negative()

Tensor.negative_

In-place version of negative()

Tensor.nelement

Alias for numel()

Tensor.nextafter

See torch.nextafter()

Tensor.nextafter_

In-place version of nextafter()

Tensor.nonzero

See torch.nonzero()

Tensor.norm

See torch.norm()

Tensor.normal_

Fills self tensor with elements samples from the normal distribution parameterized by mean and std.

Tensor.numel

See torch.numel()

Tensor.numpy

Returns the tensor as a NumPy ndarray.

Tensor.orgqr

See torch.orgqr()

Tensor.ormqr

See torch.ormqr()

Tensor.outer

See torch.outer().

Tensor.permute

See torch.permute()

Tensor.pin_memory

Copies the tensor to pinned memory, if it's not already pinned.

Tensor.pinverse

See torch.pinverse()

Tensor.polygamma

See torch.polygamma()

Tensor.polygamma_

In-place version of polygamma()

Tensor.positive

See torch.positive()

Tensor.pow

See torch.pow()

Tensor.pow_

In-place version of pow()

Tensor.prod

See torch.prod()

Tensor.put_

Copies the elements from source into the positions specified by index.

Tensor.qr

See torch.qr()

Tensor.qscheme

Returns the quantization scheme of a given QTensor.

Tensor.quantile

See torch.quantile()

Tensor.nanquantile

See torch.nanquantile()

Tensor.q_scale

Given a Tensor quantized by linear(affine) quantization, returns the scale of the underlying quantizer().

Tensor.q_zero_point

Given a Tensor quantized by linear(affine) quantization, returns the zero_point of the underlying quantizer().

Tensor.q_per_channel_scales

Given a Tensor quantized by linear (affine) per-channel quantization, returns a Tensor of scales of the underlying quantizer.

Tensor.q_per_channel_zero_points

Given a Tensor quantized by linear (affine) per-channel quantization, returns a tensor of zero_points of the underlying quantizer.

Tensor.q_per_channel_axis

Given a Tensor quantized by linear (affine) per-channel quantization, returns the index of dimension on which per-channel quantization is applied.

Tensor.rad2deg

See torch.rad2deg()

Tensor.random_

Fills self tensor with numbers sampled from the discrete uniform distribution over [from, to - 1].

Tensor.ravel

see torch.ravel()

Tensor.reciprocal

See torch.reciprocal()

Tensor.reciprocal_

In-place version of reciprocal()

Tensor.record_stream

Ensures that the tensor memory is not reused for another tensor until all current work queued on stream are complete.

Tensor.register_hook

Registers a backward hook.

Tensor.remainder

See torch.remainder()

Tensor.remainder_

In-place version of remainder()

Tensor.renorm

See torch.renorm()

Tensor.renorm_

In-place version of renorm()

Tensor.repeat

Repeats this tensor along the specified dimensions.

Tensor.repeat_interleave

See torch.repeat_interleave().

Tensor.requires_grad

Is True if gradients need to be computed for this Tensor, False otherwise.

Tensor.requires_grad_

Change if autograd should record operations on this tensor: sets this tensor's requires_grad attribute in-place.

Tensor.reshape

Returns a tensor with the same data and number of elements as self but with the specified shape.

Tensor.reshape_as

Returns this tensor as the same shape as other.

Tensor.resize_

Resizes self tensor to the specified size.

Tensor.resize_as_

Resizes the self tensor to be the same size as the specified tensor.

Tensor.retain_grad

Enables this Tensor to have their grad populated during backward().

Tensor.retains_grad

Is True if this Tensor is non-leaf and its grad is enabled to be populated during backward(), False otherwise.

Tensor.roll

See torch.roll()

Tensor.rot90

See torch.rot90()

Tensor.round

See torch.round()

Tensor.round_

In-place version of round()

Tensor.rsqrt

See torch.rsqrt()

Tensor.rsqrt_

In-place version of rsqrt()

Tensor.scatter

Out-of-place version of torch.Tensor.scatter_()

Tensor.scatter_

Writes all values from the tensor src into self at the indices specified in the index tensor.

Tensor.scatter_add_

Adds all values from the tensor src into self at the indices specified in the index tensor in a similar fashion as scatter_().

Tensor.scatter_add

Out-of-place version of torch.Tensor.scatter_add_()

Tensor.scatter_reduce_

Reduces all values from the src tensor to the indices specified in the index tensor in the self tensor using the applied reduction defined via the reduce argument ("sum", "prod", "mean", "amax", "amin").

Tensor.scatter_reduce

Out-of-place version of torch.Tensor.scatter_reduce_()

Tensor.select

See torch.select()

Tensor.select_scatter

See torch.select_scatter()

Tensor.set_

Sets the underlying storage, size, and strides.

Tensor.share_memory_

Moves the underlying storage to shared memory.

Tensor.short

self.short() is equivalent to self.to(torch.int16).

Tensor.sigmoid

See torch.sigmoid()

Tensor.sigmoid_

In-place version of sigmoid()

Tensor.sign

See torch.sign()

Tensor.sign_

In-place version of sign()

Tensor.signbit

See torch.signbit()

Tensor.sgn

See torch.sgn()

Tensor.sgn_

In-place version of sgn()

Tensor.sin

See torch.sin()

Tensor.sin_

In-place version of sin()

Tensor.sinc

See torch.sinc()

Tensor.sinc_

In-place version of sinc()

Tensor.sinh

See torch.sinh()

Tensor.sinh_

In-place version of sinh()

Tensor.asinh

See torch.asinh()

Tensor.asinh_

In-place version of asinh()

Tensor.arcsinh

See torch.arcsinh()

Tensor.arcsinh_

In-place version of arcsinh()

Tensor.size

Returns the size of the self tensor.

Tensor.slogdet

See torch.slogdet()

Tensor.slice_scatter

See torch.slice_scatter()

Tensor.sort

See torch.sort()

Tensor.split

See torch.split()

Tensor.sparse_mask

Returns a new sparse tensor with values from a strided tensor self filtered by the indices of the sparse tensor mask.

Tensor.sparse_dim

Return the number of sparse dimensions in a sparse tensor self.

Tensor.sqrt

See torch.sqrt()

Tensor.sqrt_

In-place version of sqrt()

Tensor.square

See torch.square()

Tensor.square_

In-place version of square()

Tensor.squeeze

See torch.squeeze()

Tensor.squeeze_

In-place version of squeeze()

Tensor.std

See torch.std()

Tensor.stft

See torch.stft()

Tensor.storage

Returns the underlying storage.

Tensor.storage_offset

Returns self tensor's offset in the underlying storage in terms of number of storage elements (not bytes).

Tensor.storage_type

Returns the type of the underlying storage.

Tensor.stride

Returns the stride of self tensor.

Tensor.sub

See torch.sub().

Tensor.sub_

In-place version of sub()

Tensor.subtract

See torch.subtract().

Tensor.subtract_

In-place version of subtract().

Tensor.sum

See torch.sum()

Tensor.sum_to_size

Sum this tensor to size.

Tensor.svd

See torch.svd()

Tensor.swapaxes

See torch.swapaxes()

Tensor.swapdims

See torch.swapdims()

Tensor.symeig

See torch.symeig()

Tensor.t

See torch.t()

Tensor.t_

In-place version of t()

Tensor.tensor_split

See torch.tensor_split()

Tensor.tile

See torch.tile()

Tensor.to

Performs Tensor dtype and/or device conversion.

Tensor.to_mkldnn

Returns a copy of the tensor in torch.mkldnn layout.

Tensor.take

See torch.take()

Tensor.take_along_dim

See torch.take_along_dim()

Tensor.tan

See torch.tan()

Tensor.tan_

In-place version of tan()

Tensor.tanh

See torch.tanh()

Tensor.tanh_

In-place version of tanh()

Tensor.atanh

See torch.atanh()

Tensor.atanh_

In-place version of atanh()

Tensor.arctanh

See torch.arctanh()

Tensor.arctanh_

In-place version of arctanh()

Tensor.tolist

Returns the tensor as a (nested) list.

Tensor.topk

See torch.topk()

Tensor.to_dense

Creates a strided copy of self if self is not a strided tensor, otherwise returns self.

Tensor.to_sparse

Returns a sparse copy of the tensor.

Tensor.to_sparse_csr

Convert a tensor to compressed row storage format (CSR).

Tensor.to_sparse_csc

Convert a tensor to compressed column storage (CSC) format.

Tensor.to_sparse_bsr

Convert a CSR tensor to a block sparse row (BSR) storage format of given blocksize.

Tensor.to_sparse_bsc

Convert a CSR tensor to a block sparse column (BSC) storage format of given blocksize.

Tensor.trace

See torch.trace()

Tensor.transpose

See torch.transpose()

Tensor.transpose_

In-place version of transpose()

Tensor.triangular_solve

See torch.triangular_solve()

Tensor.tril

See torch.tril()

Tensor.tril_

In-place version of tril()

Tensor.triu

See torch.triu()

Tensor.triu_

In-place version of triu()

Tensor.true_divide

See torch.true_divide()

Tensor.true_divide_

In-place version of true_divide_()

Tensor.trunc

See torch.trunc()

Tensor.trunc_

In-place version of trunc()

### Tensor.type

Last updated: 2023-01-30, 18:27

```python
Tensor.type(dtype=None, non_blocking=False, **kwargs) → str or Tensor
```

如果未提供 `dtype`，就返回类型；返回将该张量转换为指定 `dtype`。

如果张量类型与提供的 `dtype` 一致，则不复制，直接返回原张量。

**参数：**

- **dtype** (dtype or string) – 期望类型
- **non_blocking** (`bool`) – `True` 且原张量位于 pinned 内存，而目标位于 GPU（或相反情况），则异步执行复制。其它情况该参数无效。 
- ****kwargs** – 兼容，可能包含 `async` 代替 `non_blocking`， `async` 参数已启用。

Tensor.type_as

Returns this tensor cast to the type of the given tensor.

Tensor.unbind

See torch.unbind()

Tensor.unflatten

See torch.unflatten().

Tensor.unfold

Returns a view of the original tensor which contains all slices of size size from self tensor in the dimension dimension.

Tensor.uniform_

Fills self tensor with numbers sampled from the continuous uniform distribution:

Tensor.unique

Returns the unique elements of the input tensor.

Tensor.unique_consecutive

Eliminates all but the first element from every consecutive group of equivalent elements.

Tensor.unsqueeze

See torch.unsqueeze()

Tensor.unsqueeze_

In-place version of unsqueeze()

Tensor.values

Return the values tensor of a sparse COO tensor.

Tensor.var

See torch.var()

Tensor.vdot

See torch.vdot()

Tensor.view

Returns a new tensor with the same data as the self tensor but of a different shape.

Tensor.view_as

View this tensor as the same size as other.

Tensor.vsplit

See torch.vsplit()

Tensor.where

self.where(condition, y) is equivalent to torch.where(condition, self, y).

Tensor.xlogy

See torch.xlogy()

Tensor.xlogy_

In-place version of xlogy()

Tensor.zero_

Fills self tensor with zeros.

## 参考

- https://pytorch.org/docs/stable/tensors.html
