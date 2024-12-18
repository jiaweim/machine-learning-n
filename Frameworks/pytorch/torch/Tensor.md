# torch.Tensor

2022-12-13 ⭐
@author Jiawei Mao
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

## 属性

`Tensor` 主要有 8 大属性：

| 属性                                  | 说明               |
| ------------------------------------- | ------------------ |
| data                                  | 多维数组，包含数据 |
| dtype                                 | 数据类型           |
| shape                                 | 多维数组形状       |
| device                                | 张量所在设备       |
| grad, grad_fn, is_leaf, requires_grad | 计算梯度所需       |

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





## Tensor 参考

### Tensor.new_tensor

```python
Tensor.new_tensor(data, *, 
    dtype=None, 
    device=None, 
    requires_grad=False, 
    layout=torch.strided, 
    pin_memory=False) → Tensor
```



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

### numpy

> 2024-10-24 ⭐

```python
Tensor.numpy(*, force=False) → numpy.ndarray
```

返回张量的 numpy `ndarray` 形式。

如果 `force` 为 `False`（默认），则要求 tensor 位于 CPU、不需要 grad、没有设置耦合位，dtype 和 layout  NumPy 支持才执行转换。返回的 `ndarray` 和 tensor 共享内存，因此张量和 `ndarray` 变化同步。

如果 `force` 为 `True`，则等价于 `t.detach().cpu().resolve_conj().resolve_neg().numpy()`。如果 tensor 不在 CPU，或者设置了共轭位或负位，则张量与 `ndarray` 不共享内存。将 `force` 设置为 `True` 是一种获得张量 `ndarray` 形式的简单方法。

即设置 `force=True` 返回的 ndarray 可以与张量不共享内存。

参数：

- **force** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – `True` 表示 `ndarray` 可能是张量副本，而总是共享内存，默认 `False`

### scatter_

```python
Tensor.scatter_(dim, 
        index, 
        src, 
        reduce=None) → Tensor
```

按照索引张量 `index` 指定的位置用 `src` 张量的值替换 `self` 的部分值。

`src` 值的选择：当 `dimension != dim` 索引值同 `src` ；当 `dimension = dim` 索引由 `index` 中的对应值指定。

对 3D 张量，`self` 的更新方式：

- `index` 和 `src` 的索引一一对应
- `index` 的值对应 `self` 对应维度的索引

```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

该操作为 `gather()` 的逆操作。

需要注意：

- `self`, `index` 和 `src` 的 dim-count  必须相同；
- 对所有维度 `d`，要求 `index.size(d) <= src.size(d)`；
- 对所有 `d != dim` 的维度，要求 `index.size(d) <= self.size(d)`；
- `index` 和 `src` 不广播

此外，对 `gather()`，`index` 的值必须在 0 到 `self.size(dim)-1` (inclusive) 之间。

> [!WARNING]
> 当索引不是 unique (为 self 同一个位置重复赋值)，其行为不确定并导致梯度不正确，即梯度将传递到 src 中同一索引的所有位置。
> 仅对 `src.shape == index.shape` 实现反向传播。

`reduce` 用于指定降维运算。即将 `index` 指定的 `src` 中所有元素通过降维运算合并到 `self` 对应位置。

对 3D 张量，使用乘法进行降维，`self` 更新方式：

```python
self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2
```

加法降维等价于 `scatter_add_()`。

> [!WARNING]
> `reduce` 参数已弃用。使用 `scatter_reduce_()` 选项更丰富。

**参数：**

- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 索引维度
- **index** (*LongTensor*) – 待分配元素的索引，可以为空或与 `src` 相同维度。为空时，直接返回 `self`
- **src** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 提供值的张量

**关键字参数：**

- **reduce** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – 引用的缩减操作，可以为 `'add'` or `'multiply'`.

```python
>>> src = torch.arange(1, 11).reshape((2, 5)) # (2,5)
>>> src
tensor([[ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10]])
>>> index = torch.tensor([[0, 1, 2, 0]]) # (1,4)
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src) # (3,5)
tensor([[1, 0, 0, 4, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 3, 0, 0]])
# self[index[0,0],0]=self[0,0]=src[0,0]=1
# self[index[0,1],1]=self[1,1]=src[0,1]=2
# self[index[0,2],2]=self[2,2]=src[0,2]=3
# self[index[0,3],3]=self[0,3]=src[0,3]=4
>>> index = torch.tensor([[0, 1, 2], [0, 1, 4]]) # (2,3)
>>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src) # (3,5)
tensor([[1, 2, 3, 0, 0],
        [6, 7, 0, 0, 8],
        [0, 0, 0, 0, 0]])
# self[0,index[0,0]]=self[0,0]=src[0,0]=1
# self[0,index[0,1]]=self[0,1]=src[0,1]=2
# self[0,index[0,2]]=self[0,2]=src[0,2]=3
# self[1,index[1,0]]=self[1,0]=src[1,0]=6
# self[1,index[1,1]]=self[1,1]=src[1,1]=7
# self[1,index[1,2]]=self[1,4]=src[1,2]=8
>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='multiply')
tensor([[2.0000, 2.0000, 2.4600, 2.0000],
        [2.0000, 2.0000, 2.0000, 2.4600]])
>>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
...            1.23, reduce='add')
tensor([[2.0000, 2.0000, 3.2300, 2.0000],
        [2.0000, 2.0000, 2.0000, 3.2300]])
```

```python
scatter_(dim, 
        index, 
        value, *, 
        reduce=None) → Tensor:
```

同上，替换值全部指定为 `value`。

**参数：**

- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 索引维度
- **index** (*LongTensor*) – 索引，可以为空或与 `src` 同 dim-count。为空时返回 `self`.
- **value** (*Scalar*) – 替换值

**关键字参数：**

- **reduce** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – 约简操作，可以为 `'add'` or `'multiply'`.

```python
>>> index = torch.tensor([[0, 1]])
>>> value = 2
>>> torch.zeros(3, 5).scatter_(0, index, value)
tensor([[2., 0., 0., 0., 0.],
        [0., 2., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
```

### Tensor.to

```python
Tensor.to(*args, **kwargs) → Tensor
```

张量 `dtype` and/or `device` 转换。`torch.dtype` 和 `torch.device` 根据 `self.to(*args, **kwargs)` 推断。

> **NOTE**
> 如果 `self` 张量的 `torch.dtype` 和 `torch.device` 已经正确，则直接返回 `self`。否则以指定 `torch.dtype` 和 `torch.device` 返回 `self` 的副本。

下面是调用 `to` 的方法：

```python
to(dtype, 
    non_blocking=False, 
    copy=False, 
    memory_format=torch.preserve_format) → Tensor
```

返回指定 `dtype` 的张量。

```python
torch.to(device=None, 
    dtype=None, 
    non_blocking=False, 
    copy=False, 
    memory_format=torch.preserve_format) → Tensor
```

返回指定 `device` 和 `dtype`（可选）的张量。如果 `dtype` 为 `None`，则使用 `self.dtype`。当 `non_blocking`,则尝试与主机异步转换，例如，将锁业内存 CPU 张量转换为 CUDA 张量。当 `copy`，即使 `self` 已经满足需求，依然会创建一个新的张量。

```python
torch.to(other, 
    non_blocking=False, 
    copy=False) → Tensor
```

创建一个与 `other` 具有相同 `device` 和 `dtype` 的张量。

**示例：**

```python
>>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
>>> tensor.to(torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64)

>>> cuda0 = torch.device('cuda:0')
>>> tensor.to(cuda0)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], device='cuda:0')

>>> tensor.to(cuda0, dtype=torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

>>> other = torch.randn((), dtype=torch.float64, device=cuda0)
>>> tensor.to(other, non_blocking=True)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
```

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

### Tensor.grad

> 2024-10-24⭐

```python
Tensor.grad
```

该属性默认为 `None`，在第一次调用 `backward()` 计算梯度时变为张量，保存计算出的梯度，下次调用 `backward()` 将把梯度累计 (add) 起来。

与输入 `data` 的 shape 一致。


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

### Tensor.clone

参考 `torch.clone()`

### Tensor.contiguous

```python
Tensor.contiguous(memory_format=torch.contiguous_format) → Tensor
```

返回一个内存连续张量，其数据与 `self` 相同。如果 `self` 已经满足指定内存格式，则直接返回 `self`。

**参数：**

- **memory_format** (`torch.memory_format`, optional) – 指定张量的内存格式。默认: `torch.contiguous_format`。

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

### Tensor.detach

> 2024-10-24⭐

```python
Tensor.detach()
```

从当前图中分离出来，创建一个新的张量。

返回的张量不需要梯度。

该方法会影响正向模式的 AD 梯度，并且结果也不会有正向模式的 AD 梯度。

> [!NOTE]
> 返回的张量与原张量共享内存。对返回张量执行的原地操作，如 `resize_`, `resize_as_`, `set_`, `transpose_` 会触发错误。

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

### Tensor.float

```python
Tensor.float(memory_format=torch.preserve_format) → Tensor
```

`self.float()` 等价于 `self.to(torch.float32)`。

**参数：**

- **memory_format** (`torch.memory_format`, optional) – 返回张量的内存格式。默认: `torch.preserve_format`。

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

### Tensor.is_leaf

> 2024-10-24⭐

```python
Tensor.is_leaf
```

按照惯例，所有 `requires_grad`  为 `False` 的张量都是 leaf 张量。

对 `requires_grad` 为 `True` 的张量，如果它们是由用户创建的，则为 leaf 张量。这意味着它们不是操作的结果，因此 `grad_fn` 为 `None`。

只有 leaf 张量在调用 `backward()` 后保留 `grad` 值。非 leaf 张量，需要调用 `retain_grad()` 才会保留其 `grad` 值。

```python
>>> a = torch.rand(10, requires_grad=True)
>>> a.is_leaf
True
>>> b = torch.rand(10, requires_grad=True).cuda()
>>> b.is_leaf
False
# b 是通过将 cpu 张量转换为 cuda 张量的操作创建的
>>> c = torch.rand(10, requires_grad=True) + 2
>>> c.is_leaf
False
# c 是通过加法操作创建的
>>> d = torch.rand(10).cuda()
>>> d.is_leaf
True
# d 不需要梯度，因此没有创建它的操作（由 autograd 跟踪的操作）
>>> e = torch.rand(10).cuda().requires_grad_()
>>> e.is_leaf
True
# e 需要梯度，且没有创建它的操作
>>> f = torch.rand(10, requires_grad=True, device="cuda")
>>> f.is_leaf
True
# f 要求梯度，且没有创建它的操作
```


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

### Tensor.item

```python
Tensor.item() → number
```

对只包含一个元素的张量，以 Python 数字返回其元素值。对包含多个元素的张量，用 `tolist()`。

该操作不可微。

**示例：**

```python
>>> x = torch.tensor([1.0])
>>> x.item()
1.0
```

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

### Tensor.numel

等价于 `torch.numel()`。

返回张量所含元素总数。

例如：

```python
>>> a = torch.randn(1, 2, 3, 4, 5)
>>> torch.numel(a)
120
>>> a = torch.zeros(4,4)
>>> torch.numel(a)
16
```

### Tensor.numpy

> 2024-10-24⭐

```python
Tensor.numpy(*, force=False) → numpy.ndarray
```

将 tensor 转换为 NumPy `ndarray`。

`force=False` 时（默认），仅当张量在 CPU 上、不需要梯度、没有共轭 bit set，并且是 NumPy 支持的 dtype 和 layout 才转换。返回的 ndarray 和张量共享内存。

`force=True` 等价于 `t.detach().cpu().resolve_conj().resolve_neg().numpy()`。如果张量不在 CPU，或设置了 conjugate 或 negative bit，则张量不与返回的 ndarray 共享内存。

**参数：**

- **force** (`bool`) `True` 时 `ndarray` 可能是张量的副本，不一定共享内存。

Tensor.orgqr

See torch.orgqr()

Tensor.ormqr

See torch.ormqr()

Tensor.outer

See torch.outer().

### Tensor.permute

```python
Tensor.permute(*dims) → Tensor
```

参考 `torch.permute()`。

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

### Tensor.reshape

> 2024年10月22日 ⭐

返回数据相同但 shape 不同的 tensor。如果新的 `shape` 与当前 `shape` 兼容，则返回一个 view。

等同于 `torch.reshape()`。

Tensor.reshape_as

Returns this tensor as the same shape as other.

Tensor.resize_

Resizes self tensor to the specified size.

Tensor.resize_as_

Resizes the self tensor to be the same size as the specified tensor.

Tensor.retain_grad

Enables this Tensor to have their grad populated during backward().

### Tensor.retains_grad

> 2024-10-24⭐

```python
Tensor.retains_grad
```

如果该张量为 non-leaf，且其 `grad` 在 `backward()` 期间保存，则为 `True`。否则为 `False`。


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

### Tensor.shape

> 2024年10月22日 ⭐

返回 tensor 的 size，同 `size`。

```python
>>> t = torch.empty(3, 4, 5)
>>> t.size()
torch.Size([3, 4, 5])
>>> t.shape
torch.Size([3, 4, 5])
```

### Tensor.size

```python
Tensor.size(dim=None) → torch.Size or int
```

返回 `self` 张量的大小：

- 不指定 `dim`，返回 `torch.Size`，它是 `tuple` 的子类；
- 指定 `dim`，返回 `int`，为该维度的大小 

**参数：**

- **dim** (`int`, optional)

要检索的维度。

**示例：**

```python
>>> t = torch.empty(3, 4, 5)
>>> t.size()
torch.Size([3, 4, 5])
>>> t.size(dim=1)
4
```

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

### Tensor.tolist

```python
Tensor.tolist() → list or number
```

以 list 形式返回张量值；对标量直接返回一个 Python 数，同 `item()`。如果需要，自动将张量移到 CPU。

该操作不可微。

**示例：**

```python
>>> a = torch.randn(2, 2)
>>> a.tolist()
[[0.012766935862600803, 0.5415473580360413],
 [-0.08909505605697632, 0.7729271650314331]]
>>> a[0,0].tolist()
0.012766935862600803
```

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

### Tensor.view

```python
Tensor.view(*shape) → Tensor
```

返回一个与 `self` 张量数据相同但 `shape` 不同的新张量。

返回的张量共享数据，元素个数相同，但是 size 可能不同。对 view 张量，其 size 必须与原张量的 size 和 stride 兼容，

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
