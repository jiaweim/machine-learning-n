# torch

Last updated: 2023-02-13, 10:24
***

## Tensors

|操作|说明|
|---|---|
|`is_tensor`|`obj` 是否为 PyTorch tensor|
|is_storage|`obj` 是否为 PyTorch storage 对象|
|is_complex|`input` 的数据类型是否为复数类型，即 `torch.complex64` 和 `torch.complex128`|
|is_conj|`input` 是否为共轭张量，即其 conjugate bit 是否设为 True|
|is_floating_point|`input` 数据类型是否为浮点数，包括 `torch.float64`, `torch.float32`, `torch.float16` 和 `torch.bfloat16`|
|is_nonzero|`input` 是否是不等于零的单元素张量|
|set_default_dtype|将默认浮点类型设为 `d`|
|get_default_dtype|返回当前默认浮点 `torch.dtype`|
|set_default_tensor_type|将 `torch.Tensor` 默认浮点类型设置为 `t`|
|numel|`input` 张量元素总数|
|set_printoptions|设置打印选项|
|set_flush_denormal|禁用 CPU 上的非正规浮点数|

## 创建张量

### torch.tensor

> 2024年10月22日 ⭐

```python
torch.tensor(data, *, 
    dtype=None, 
    device=None, 
    requires_grad=False, 
    pin_memory=False) → Tensor
```

**复制** `data` 的数据创建张量，该张量没有 autograd 历史，也称为叶张量（leaf tensor）。

> [!WARNING]
> 对张量类参数，建议使用 `torch.Tensor.clone()`, `torch.Tensor.detach()` 和 `torch.Tensor.requires_grad_()`。设 `t` 为张量，则 `torch.tensor(t)` 等价于 `t.clone().detach()`；而 `torch.tensor(t, requires_grad=True)` 等价于 `t.clone().detach().requires_grad_(True)`。

**参数：**

- **data** (`array_like`) - 张量的初始化数据。支持 `list`, `tuple`, `ndarray`, scalar 等类型。

**关键字参数：**

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 设置张量类型。`None` 表示从 `data` 推断类型。
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。`None` 且 `data` 为 tensor，则使用 `data` 所在 device。`None` 且 `data` 不是 tensor，则使用当前 device。
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – autograd 是否记录返回的 tensor 上的操作。默认 `False`
- **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否将返回的张量放在 pinned memory，仅用于 cpu 张量。默认 `False`.

```python
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]) # 从 list 创建张量
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])  # 根据数据推断类型
tensor([ 0,  1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
...              dtype=torch.float64,
...              device=torch.device('cuda:0'))  # creates a double tensor on a CUDA device
tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # 创建 0 维张量：标量
tensor(3.1416)

>>> torch.tensor([])  # 创建空张量: size (0,)
tensor([])
```

sparse_coo_tensor

Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.

asarray

Converts obj to a tensor.

### torch.as_tensor

```python
torch.as_tensor(data, dtype=None, device=None) → Tensor
```

将 `data` 转换为张量，尽可能**共享内存**并保留 autograd 历史。

- 如果 `data` 已是张量
  - dtype 和 device 满足要求，则直接返回 `data`；
  - dtype 和 device 不符合要求，则返回副本，类似 `data.to(dtype=dtype, device=device)`；
- 如果 `data` 是 ndarray，且 dtype 和 device 满足要求，则使用 `torch.from_numpy()` 创建张量

> **NOTE**
> `torch.tensor()` 永远不会共享数据。

**参数：**

- **data** (`array_like`) – 初始数据，支持 list, tuple, NumPy ndarray, scalar 等类型。
- **dtype** (`torch.dtype`, optional) – 张量类型，默认 `None` 表示从数据推断类型。
- **device** (`torch.device`, optional) – `None` 且 `data` 是张量，则直接使用 `data` device；`None` 且 `data` 不是张量，则为 CPU。

**示例：**

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1 # 与 ndarray 共享内存
>>> a
array([-1,  2,  3])

>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a, device=torch.device('cuda'))
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([1,  2,  3])
```

as_strided

Create a view of an existing torch.Tensor input with specified size, stride and storage_offset.

### torch.from_numpy

> 2024-10-24⭐

```python
torch.from_numpy(ndarray) → Tensor
```

使用 `numpy.ndarray` 创建 `Tensor`。

返回的张量与 `ndarray` 共享内存。修改张量 `ndarray` 随之改变，反之亦然。返回的张量不能调整大小。

目前支持的 `ndarray` 类型：`numpy.float64`, `numpy.float32`, `numpy.float16`, `numpy.complex64`, `numpy.complex128`, `numpy.int64`, `numpy.int32`, `numpy.int16`, `numpy.int8`, `numpy.uint8` 和 `numpy.bool`。

> [!WARNING]
> 从 read-only `ndarray` 创建的张量不支持写入。

```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1 # 修改张量
>>> a # ndarray 随之变化
array([-1,  2,  3]) 
```

from_dlpack

Converts a tensor from an external library into a torch.Tensor.

frombuffer

Creates a 1-dimensional Tensor from an object that implements the Python buffer protocol.

### torch.zeros

> 2024-10-24⭐

```python
torch.zeros(*size, *, 
    out=None, 
    dtype=None, 
    layout=torch.strided, 
    device=None, 
    requires_grad=False
) → Tensor
```

创建以 0 填充的张量，其 shape 由参数 `size` 定义。

**参数：**

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 定义输出张量 shape 的整数序列。支持可变参数、list 和 tuple。

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 期望张量类型。默认：`None` 表示使用全局默认类型 (参考 [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 指定张量 layout。默认 `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 指定张量 device。默认：`None` 表示使用默认张量类型的当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 对 CPU 张量类型为 CPU，对 CUDA 张量类型为当前 CUDA device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录返回张量上的操作。默认 `False`.

```python
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

>>> torch.zeros(5)
tensor([ 0.,  0.,  0.,  0.,  0.])
```

### torch.zeros_like

> 2024-10-24⭐

```python
torch.zeros_like(input, *, 
    dtype=None, 
    layout=None, 
    device=None, 
    requires_grad=False, 
    memory_format=torch.preserve_format) → Tensor
```

创建以 0 填充的张量，其 shape 与 `input` 相同。`torch.zeros_like(input)` 等价于 `torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。

> [!WARNING]
> 从 0.4 开始，此函数不再支持 `out` 关键字。作为替代方案，旧版 `torch.zeros_like(input, out=output)` 相当于 `torch.zeros(input.size(), out=output)`.

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `input` 的 size 决定输出张量的 size

**关键字参数：**

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量类型。默认: `None` 表示采用 `input` 的 dtype
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 指定张量的 layout。默认：`None` 表示采用 `input` 的 layout
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 指定张量的 device。默认：`None` 表示 `input` 的 device
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录张量上的操作。默认 `False`.
- **memory_format** ([`torch.memory_format`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format), optional) – 指定张量的内存格式。默认 `torch.preserve_format`.

```python
>>> input = torch.empty(2, 3)
>>> torch.zeros_like(input)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```

### torch.ones

> 2024-10-24⭐

```python
torch.ones(*size, *, 
    out=None, 
    dtype=None, 
    layout=torch.strided, 
    device=None, 
    requires_grad=False) → Tensor
```

创建以 1 填充的张量，其 shape 由参数 `size` 定义。

**参数：**

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 定义输出张量 shape 的整数序列。支持可变参数、list 和 tuple。

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 期望张量类型。默认：`None` 表示使用全局默认类型 (参考 [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 指定张量 layout。默认 `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 指定张量 device。默认：`None` 表示使用默认张量类型的当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 对 CPU 张量类型为 CPU，对 CUDA 张量类型为当前 CUDA device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录返回张量上的操作。默认 `False`.

```python
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

>>> torch.ones(5)
tensor([ 1.,  1.,  1.,  1.,  1.])
```

### torch.ones_like

> 2024-10-24⭐

```python
torch.ones_like(input, *, 
    dtype=None, 
    layout=None, 
    device=None, 
    requires_grad=False, 
    memory_format=torch.preserve_format) → Tensor
```

创建以 1 填充的张量，其 shape 与 `input` 相同。`torch.ones_like(input)` 等价于 `torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。

> [!WARNING]
> 从 0.4 开始，此函数不再支持 `out` 关键字。作为替代方案，旧版 `torch.ones_like(input, out=output)` 相当于 `torch.ones(input.size(), out=output)`.

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `input` 的 size 决定输出张量的 size

**关键字参数：**

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 指定张量类型。默认: `None` 表示采用 `input` 的 dtype
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 指定张量 layout。默认：`None` 表示采用 `input` 的 layout
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 指定张量 device。默认：`None` 表示 `input` 的 device
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录张量上的操作。默认 `False`.
- **memory_format** ([`torch.memory_format`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format), optional) – 指定张量的内存格式。默认 `torch.preserve_format`.

```python
>>> input = torch.empty(2, 3)
>>> torch.ones_like(input)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
```

### arange

> 2024-10-24⭐

```python
torch.arange(start=0, 
        end, 
        step=1, *, 
        out=None, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False) → Tensor
```

创建一个 1D tensor，即在区间 `[start, end)` 以步长 `step` 创建大小为 $\lceil\frac{\text{end}-\text{start}}{\text{step}}\rceil$ 的 tensor。

如果 `step` 为非整数，在与 `end` 比较时会有浮点数 round-error 问题；为了避免该问题，建议从 `end` 中减去一个 epsilon 值。
$$
\text{out}_{i+1}=\text{out}_i+\text{step}
$$
**参数：**

- `start` (Number): 起点值，默认 0
- `end` (Number)：重点值
- `step` (Number): 步长，默认 1 

**关键字参数：**

- **out** (`Tensor`, optional)：输出张量。
- **dtype** (`torch.dtype`, optional)：tensor 类型。默认：如果为 `None`，则使用全局默认，即 `torch.set_default_dtype()`，如果不指定 `dtype`，从输入参数推断类型。如果 `start`, `end` 或 `step` 任意值为 float，则 `dtype` 为默认 `dtype`，参考 `get_default_dtype()`。否则 `dtype` 为 `torch.int64`。
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 返回 tensor 的 layout。默认 `torch.strided`
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) –  设置 device。`None` 表示使用当前设备，参考 `torch.set_default_device()`。`device` 对 CPU-tensor 为 CPU，对 CUDA-tensor 为 CUDA。
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – autograd 需要记录 tensor 上的操作，默认  `False`.

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```

### range

> 2024-10-24⭐

```python
torch.range(start=0, 
        end, 
        step=1, *, 
        out=None, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False) → Tensor
```

> [!WARNING]
> 已弃用。该函数与 Python 内置的 `range` 不一致，改用 [torch.arange](#arange)，它会生成 [start, end) 范围的值。

### linspace

> 2024-10-24⭐

```python
torch.linspace(start, 
        end, 
        steps, *, 
        out=None, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False) → Tensor
```

创建一个长度为 `steps` 的 1D 张量，张量值从 `start` 到 `end` （inclusive）等距分布。即：

$$
(\text{start},\text{start}+\frac{\text{end}-\text{start}}{\text{steps}-1},\cdots,\text{start}+(\text{steps}-2)*\frac{\text{end}-\text{start}}{\text{steps}-1},\text{end})
$$

从 PyTorch 1.11 开始，`linspace` 需要 `steps` 参数。使用 `steps=100` 重现之前版本的行为。 

**参数：**

- **start** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 起始值，如果是 `Tensor`，必须是 0D
- **end** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 末尾值，如果使用 `Tensor`，必须是 0D
- **steps** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 张量 size

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([*torch.dtype*](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,* *optional*) – 数据类型。默认：`None` 表示当 `start` 和 `end` 都是实数时使用全局默认 dtype (see torch.get_default_dtype())，当其中一个为复数，则使用相应的复数 dtype
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 对默认张量类型使用当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量为 cpu，对 cuda 张量为当前 cuda device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 时候记录操作。默认: `False`.

```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])
```

### logspace

> 2024-10-24⭐

```python
torch.logspace(start, 
        end, 
        steps, 
        base=10.0, *, 
        out=None, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False) → Tensor
```

创建 size 为 `steps` 的张量，其值在 $\text{base}^{\text{start}}$ 到 $\text{base}^{\text{end}}$ (inclusive) 之间以底数为 `base` 的对数均匀分布。即：

$$
(\text{base}^{\text{start}},\text{base}^{(\text{start}+\frac{\text{end}-\text{start}}{\text{steps}-1})},\cdots,\text{base}^{(\text{start}+(\text{steps}-2)*\frac{\text{end}-\text{start}}{\text{steps}-1})},\text{base}^{\text{end}})
$$

从 PyTorch 1.11 开始，`logspace` 需要 `steps` 参数。使用 `steps=100` 可以恢复之前的行为。

**参数：**

- **start** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 起始值，使用 `Tensor` 必须是 0D
- **end** ([*float*](https://docs.python.org/3/library/functions.html#float) *or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 末尾值，使用 `Tensor` 必须是 0D
- **steps** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 张量 size
- **base** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – 底数，默认: `10.0`.

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([*torch.dtype*](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,* *optional*) – 数据类型。默认：`None` 表示当 `start` 和 `end` 都是实数时使用全局默认 dtype (see torch.get_default_dtype())，当其中一个为复数，则使用相应的复数 dtype
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 对默认张量类型使用当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量为 cpu，对 cuda 张量为当前 cuda device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录操作。默认: `False`.

```python
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=5)
tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
>>> torch.logspace(start=0.1, end=1.0, steps=1)
tensor([1.2589])
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])
```

### eye

> 2024-10-24⭐

```python
torch.eye(n, 
          m=None, *, 
          out=None, 
          dtype=None, 
          layout=torch.strided, 
          device=None, 
          requires_grad=False) → Tensor
```

创建一个二维张量，对角线上值为 1，其它地方为 0.

**参数：**

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 行数
- **m** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – 列数，默认为 `n`

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 张量类型。默认：`None` 表示 global 默认 (see [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 设置 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 设置 device。默认：`None` 表示对默认张量类型使用当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量类型使用 cpu，对 cuda 张量类型使用当前 cuda device。
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录操作。默认: `False`.

**返回：**

- 2D 张量，对角线上为 1，其它地方为 0.

**返回类型：**

- `Tensor`

```python
>>> torch.eye(3)
tensor([[ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]])
```

### empty

> 2024-10-24⭐

```python
torch.empty(*size, *, 
    out=None, 
    dtype=None, 
    layout=torch.strided, 
    device=None, 
    requires_grad=False, 
    pin_memory=False, 
    memory_format=torch.contiguous_format) → Tensor
```

创建一个未初始化的张量，其 shape 由 `size` 定义。

> [!NOTE]
> 如果 `torch.use_deterministic_algorithms()` 和 `torch.utils.deterministic.fill_uninitialized_memory` 都设置为 `True`，则将初始化输出张量，以防止任何可能的不确定行为将数据用作操作的输入。浮点数和 complex 张量用 NaN 填充，整数张量用其 Integet.Max 填充。

**参数：**

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 整数序列，定义张量 shape。支持可变参数和 list, tuple 等集合

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 张量类型。默认：`None` 使用 global 默认 (see [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 表示对默认张量类型使用当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量类型使用 cpu，对 cuda 张量类型使用当前 cuda device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 支持梯度。默认: `False`.
- **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – `True` 表示是否在锁页内存中分配张量。仅用于 cpu 张量，默认 `False`。
- **memory_format** ([`torch.memory_format`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format), optional) – 张量内存格式。默认: `torch.contiguous_format`.

```python
>>> torch.empty((2,3), dtype=torch.int64)
tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
        [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])
```

### empty_like

> 2024-10-24⭐

```python
torch.empty_like(input, *, 
    dtype=None, 
    layout=None, 
    device=None, 
    requires_grad=False, 
    memory_format=torch.preserve_format) → Tensor
```

参考 [zeros_like](#torchzeros_like)。

返回一个 `size` 与 `input` 相同的张量，值未初始化。`torch.empty_like(input)` 等价于 `torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。

**参数：**

- **input** (`Tensor`)

`input` 的 size 决定输出张量的 size。

**例如：**

```python
>>> a=torch.empty((2,3), dtype=torch.int32, device = 'cuda')
>>> torch.empty_like(a)
tensor([[0, 0, 0],
        [0, 0, 0]], device='cuda:0', dtype=torch.int32)
```

### empty_strided

> 2024-10-24⭐

```python
torch.empty_strided(size, 
                    stride, *, 
                    dtype=None, 
                    layout=None, 
                    device=None, 
                    requires_grad=False, 
                    pin_memory=False) → Tensor
```

创建指定 `size` 和 `stride` 的张量，不初始化。

> [!WARNING]
> 如果创建的张量出现 overlapped (多个 indices 引用内存中的同一个元素)，则其行为不确定。

> [!NOTE]
> 如果 `torch.use_deterministic_algorithms()` 和 `torch.utils.deterministic.fill_uninitialized_memory` 都设置为 `True`，则将初始化输出张量，以防止任何可能的不确定行为将数据用作操作的输入。浮点数和 complex 张量用 NaN 填充，整数张量用其 Integet.Max 填充。

**参数：**

- **size** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*int*](https://docs.python.org/3/library/functions.html#int)) – 张量 shape
- **stride** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*int*](https://docs.python.org/3/library/functions.html#int)) – 张量 strides

**关键字参数：**

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 张量类型。默认：`None` 使用 global 默认 (see [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 表示对默认张量类型使用当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量类型使用 cpu，对 cuda 张量类型使用当前 cuda device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 支持梯度。默认: `False`.
- **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – `True` 表示在锁业内存中分配张量。仅用于 cpu 张量，默认 `False`。

```python
>>> a = torch.empty_strided((2, 3), (1, 2))
>>> a
tensor([[8.9683e-44, 4.4842e-44, 5.1239e+07],
        [0.0000e+00, 0.0000e+00, 3.0705e-41]])
>>> a.stride()
(1, 2)
>>> a.size()
torch.Size([2, 3])
```

### full

> 2024-10-24⭐

```python
torch.full(size, 
        fill_value, *, 
        out=None, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False) → Tensor
```

创建以 `fill_value` 填充大小为 `size` 的张量。张量的 dtype 从 `fill_value` 推断出来。

**参数：**

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 定义张量 shape，支持包含整数的 list, tuple 和 [`torch.Size`](https://pytorch.org/docs/stable/size.html#torch.Size).
- **fill_value** (*Scalar*) – 用来填充张量的值.

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 张量类型，默认：`None` 表示使用 global 默认 (see [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 表示使用默认张量类型的当前 device (see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). [`device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) 对 cpu 张量类型为 cpu，对 cuda 张量类型为当前 CUDA device
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否记录操作。默认: `False`.

```python
>>> torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```

### full_like

> 2024-10-24⭐

```python
torch.full_like(input, 
        fill_value, \*, 
        dtype=None, 
        layout=torch.strided, 
        device=None, 
        requires_grad=False, 
        memory_format=torch.preserve_format) → Tensor
```

参考 [zeros_like](#torchzeros_like)。



quantize_per_tensor

Converts a float tensor to a quantized tensor with given scale and zero point.

quantize_per_channel

Converts a float tensor to a per-channel quantized tensor with given scales and zero points.

dequantize

Returns an fp32 Tensor by dequantizing a quantized Tensor

complex

Constructs a complex tensor with its real part equal to real and its imaginary part equal to imag.

polar

Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value abs and angle angle.

heaviside

Computes the Heaviside step function for each element in input.


## 索引、切片、连接和突变

|操作|说明|
|---|---|
|[reshape](#torchreshape)|返回指定形状的张量，与输入张量具有相同的数据和元素数|

### torch.adjoint

```python
torch.adjoint(Tensor) → Tensor
```

将张量最后两个维度转换，然后返回共轭视图。

对复数张量，`x.adjoint()` 等价于 `x.transpose(-2, -1).conj()`；对实数张量等价于 `x.transpose(-2, -1)`。

```python
>>> x = torch.arange(4, dtype=torch.float)
>>> A = torch.complex(x, x).reshape(2, 2)
>>> A
tensor([[0.+0.j, 1.+1.j],
        [2.+2.j, 3.+3.j]])
>>> A.adjoint()
tensor([[0.-0.j, 2.-2.j],
        [1.-1.j, 3.-3.j]])
>>> (A.adjoint() == A.mH).all()
tensor(True)
```

### torch.argwhere

```python
torch.argwhere(input) → Tensor
```

返回一个包含 `input` 中所有非零元素索引的张量。结果中，每行对应一个输入中非零元素的索引。

如果 `input` 有 n 维，则得到的索引张量 shape 为 $(z\times n)$，其中 z 是 `input` 张量非零元素个数。

> **NOTE** 该函数功能与 NumPy 的 `argwhere` 类似。

例如：

```python
>>> t = torch.tensor([1, 0, 1])
>>> torch.argwhere(t)
tensor([[0],
        [2]])
>>> t = torch.tensor([[1, 0, 1], [0, 1, 1]])
>>> torch.argwhere(t)
tensor([[0, 0],
        [0, 2],
        [1, 1],
        [1, 2]])
```

### torch.cat

> 2024年10月22日 ⭐

```python
torch.cat(tensors, dim=0, *, out=None) → Tensor
```

将张量序列 `tensors` 沿指定维度串联起来。所有张量的 shape 必须相同（除了连接的维度）或 empty。

`torch.cat()` 可以看作 `torch.split()` 和 `torch.chunk()` 的逆操作。

> [!TIP]
>
> `torch.stack()` 沿着新维度串联。

**参数：**

- **tensors** (sequence of `Tensors`) – *相同类型*的张量序列。non-empty 张量除了 cat 维度，shape 必须相同。
- **dim** (`int`, optional) – 连接的维度。

**关键字参数：**

- **out** (`Tensor`, optional) – 输出张量。

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

### torch.concat

> 2024年10月22日 ⭐

[torch.cat()](#torchcat) 的别名。

### torch.concatenate

> 2024年10月22日 ⭐

[torch.cat()](#torchcat) 的别名。

### conj

Returns a view of input with a flipped conjugate bit.

### chunk

Attempts to split a tensor into the specified number of chunks.

### dsplit

Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.

### column_stack

Creates a new tensor by horizontally stacking the tensors in tensors.

### dstack

Stack tensors in sequence depthwise (along third axis).

### gather

Gathers values along an axis specified by dim.

### hsplit

Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.

### hstack

Stack tensors in sequence horizontally (column wise).

### index_add

See index_add_() for function description.

### index_copy

See index_add_() for function description.

### index_reduce

See index_reduce_() for function description.

### index_select

Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.

### masked_select

Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.

### movedim

Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

### moveaxis

Alias for torch.movedim().

### narrow

Returns a new tensor that is a narrowed version of input tensor.

### narrow_copy

### torch.nonzero

```python
torch.nonzero(input, 
    *, 
    out=None, 
    as_tuple=False) → LongTensor or tuple of LongTensors
```



### torch.permute

```python
torch.permute(input, dims) → Tensor
```

返回 `input` 张量的一个 **view**，其维度重新排列。

**参数：**

- **input** (`Tensor`) - 输入张量。
- **dims** (tuple of python:int) - 维度顺序。

```python
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```

### reshape

Returns a tensor with the same data and number of elements as input, but with the specified shape.

### row_stack

Alias of torch.vstack().

### select

Slices the input tensor along the selected dimension at the given index.

### scatter

Out-of-place version of torch.Tensor.scatter_()

### diagonal_scatter

Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.

### select_scatter

Embeds the values of the src tensor into input at the given index.

### slice_scatter

Embeds the values of the src tensor into input at the given dimension.

### scatter_add

Out-of-place version of torch.Tensor.scatter_add_()

### scatter_reduce

Out-of-place version of torch.Tensor.scatter_reduce_()

### torch.split

```python
torch.split(tensor, split_size_or_sections, dim=0)
```

把张量分成若干块，每块都是原张量的一个视图。

若 `split_size_or_sections` 是整数，将张量等分。如果张量在 `dim` 维度不能被 `split_size` 整除，最后一个chunk 会小一点。

若 `split_size_or_sections` 是 list，则根据 `split_size_or_sections` 在 `dim` 维度将张量拆分为 `len(split_size_or_sections)` 份。

**参数：**

- **tensor** (`Tensor`) – 待拆分张量
- **split_size_or_sections** (`int`) or (`list(int)`) – 单个 chunk 大小，或包含每个 chunk 大小的 list
- **dim** (`int`) – 进行拆分的维度

**返回：**

- `List[Tensor]`

**示例：**

```python
>>> a = torch.arange(10).reshape(5,2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
>>> torch.split(a, [1,4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))
```

### torch.squeeze

```python
torch.squeeze(input, dim=None) → Tensor
```

对输入张量 `input`，删除所有大小为 1 的维度。

例如，如果 `input` 的 shape 为 (A×1×B×C×1×D)，则输出张量 shape 为 (A×B×C×D)。

当指定 `dim`，则只对该维度进行操作。如果 `input` shape 为 (A×1×B)，则 `squeeze(input, 0)` 不改变张量，而 `squeeze(input, 1)` 返回张量 shape 为 (A×B)。

> **NOTE**
> 返回的张量与 `input` 张量**共享内存**，因此改变一个张量，另一个也随之改变。

> **WARNING**
> 如果张量的 batch 维度为 1，那么 `squeeze(input)` 会删除 batch 维度，可能导致意外错误。

**参数：**

- **input** (`Tensor`) – 输入张量。
- **dim** (`int`, optional) – 如果指定，则只对指定维度进行操作。

**示例：**

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
```

### stack

Concatenates a sequence of tensors along a new dimension.

### swapaxes

Alias for torch.transpose().

### swapdims

Alias for torch.transpose().

### t

Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

### take

Returns a new tensor with the elements of input at the given indices.

### take_along_dim

Selects values from input at the 1-dimensional indices from indices along the given dim.

### tensor_split

Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.

### tile

Constructs a tensor by repeating the elements of input.

### transpose

Returns a tensor that is a transposed version of input.

### unbind

Removes a tensor dimension.

### unravel_index

[`unravel_index`](https://pytorch.org/docs/stable/generated/torch.unravel_index.html#torch.unravel_index)

### torch.unsqueeze

```python
torch.unsqueeze(input, dim) → Tensor
```

在指定位置插入一个大小为 1 的维度，返回一个新的张量。

返回的张量与原张量共享底层数据。

`dim` 值的有效范围为 `[-input.dim() - 1, input.dim() + 1)`。负数 `dim` 对应维度为 `dim = dim + input.dim() + 1`。

**参数：**

- **input** (`Tensor`) – 输入张量。
- **dim** (`int`) – 插入维度的位置索引。

**示例：**

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

### vsplit

Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.

### vstack

Stack tensors in sequence vertically (row wise).

### where

Return a tensor of elements selected from either x or y, depending on condition.

## 随机采样

seed

Sets the seed for generating random numbers to a non-deterministic random number.

manual_seed

Sets the seed for generating random numbers.

initial_seed

Returns the initial seed for generating random numbers as a Python long.

get_rng_state

Returns the random number generator state as a torch.ByteTensor.

set_rng_state

Sets the random number generator state.

torch.default_generator Returns the default CPU torch.Generator
bernoulli

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

multinomial

Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.

### normal

> 2024-10-24⭐

```python
torch.normal(mean, std, *, generator=None, out=None) → Tensor
```

对每个元素使用指定平均值和标准差从正态分布抽取随机数。

`mean` 为张量，指定每个正态分布的平均值。

`std` 为张量，给出每个正态分布的标准差。

`mean` 和 `std` 的 shape 不需要匹配，但是每个张量中的元素总数要相同。

> [!NOTE]
> 当 shape 不匹配，`mean`  的 shape 作为返回张量的 shape。
> 当 `std` 是 CUDA-tensor，该函数将其 device 与 CPU 同步。

根据 `mean` 和 `std` 取张量还是标准，有 4 种组合形式，下面依次描述。

**参数：**

- **mean** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 包含每个元素的平均值的张量
- **std** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 包含每个元素的标准差的张量

**关键字参数：**

- **generator** ([`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), optional) – 用于抽样的伪随机数生成器
- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
          8.0505,   8.1408,   9.0563,  10.0566])
```

```python
torch.normal(mean=0.0, std, *, out=None) → Tensor
```

与上面函数类似，但是所有元素共享相同的平均值。

**参数：**

- **mean** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – 所有分布的平均值
- **std** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 包含每个元素的标准差的张量

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> torch.normal(mean=0.5, std=torch.arange(1., 6.)) # 5 个元素，5 个标准差
tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])
```

```python
torch.normal(mean, std=1.0, *, out=None) → Tensor
```

同上，但所有元素共享标准差。

**参数：**

- **mean** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 包含每个元素的平均值的张量
- **std** ([*float*](https://docs.python.org/3/library/functions.html#float)*,* *optional*) – 所有分布的标准差

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> torch.normal(mean=torch.arange(1., 6.))
tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])
```

```python
torch.normal(mean, std, size, *, out=None) → Tensor
```

同上，但是所有元素共享平均值和标准差。返回张量的大小由 `size` 指定。

**参数：**

- **mean** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 所有分布的平均值
- **std** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 所有分布的标准差
- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 定义输出张量的 shape

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> torch.normal(2, 3, size=(1, 4))
tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
```

poisson

Returns a tensor of the same size as input with each element sampled from a Poisson distribution with rate parameter given by the corresponding element in input i.e.,

rand

Returns a tensor filled with random numbers from a uniform distribution on the interval 
[
0
,
1
)
[0,1)

rand_like

Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval 
[
0
,
1
)
[0,1).

randint

Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

randint_like

Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive).

randn

Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

randn_like

Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.

randperm

Returns a random permutation of integers from 0 to n - 1.

### torch.manual_seed

```python
torch.manual_seed(seed)
```

设置生成随机数的种子。返回 `torch.Generator` 对象。

**参数：**

- **seed** (`int`)

期望 seed。取值范围 [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]。负数按公式 0xffff_ffff_ffff_ffff + seed 重新映射到正数。

> **NOTE**
用随机数初始化张量是常见操作（如模型权重初始化），但有时（特别是研究中）希望确保结果的可重复性，手动设置随机数生成器的 seed 可以帮助做到这一点。

```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```

```txt
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
```

可以看到 `random1` 和 `random3` 的值相同，`random2` 和 `random4` 的值相同。手动设置 RNG 的 seed 会重置。

### rand

> 2024-10-24⭐

```python
torch.rand(*size, *, 
           generator=None, 
           out=None, 
           dtype=None, 
           layout=torch.strided, 
           device=None, 
           requires_grad=False, 
           pin_memory=False) → Tensor
```

创建张量，从均匀分布 $[0,1)$ 中随机抽样填充张量，shape 由 `size` 定义。

**参数：**

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int)*...*) – 整数序列。支持可变参数或 list, tuple 之类的集合

**关键字参数：**

- **generator** ([`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), optional) – 用于采样的伪随机数生成器
- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量
- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 张量类型。默认：`None` 表示使用全局默认 (see [`torch.set_default_dtype()`](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html#torch.set_default_dtype)).
- **layout** ([`torch.layout`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout), optional) – 张量 layout。默认: `torch.strided`.
- **device** ([`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device), optional) – 张量 device。默认：`None` 表示对默认张量类型使用当前 device(see [`torch.set_default_device()`](https://pytorch.org/docs/stable/generated/torch.set_default_device.html#torch.set_default_device)). 对 cpu 张量类型使用 cpu，对 cuda 张量类型使用当前 cuda device.
- **requires_grad** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 支持梯度。默认: `False`.
- **pin_memory** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – `True` 表示是否在锁页内存中分配张量。仅用于 cpu 张量，默认 `False`。

```python
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
```

### rand_like

> 2024-10-24⭐

```python
torch.rand_like(input, *, 
                dtype=None, 
                layout=None, 
                device=None, 
                requires_grad=False, 
                memory_format=torch.preserve_format) → Tensor
```

`torch.rand_like(input)` 等价于 `torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`。参考 [rand](#rand)。

### randint

```python
torch.randint(low=0, 
              high, 
              size, \*, 
              generator=None, 
              out=None, 
              dtype=None, 
              layout=torch.strided, 
              device=None, 
              requires_grad=False) → Tensor
```

生成张量，使用 `low` (inclusive) 和 `high` (exclusive) 之间均分生成的随机整数填充，其 shape 由 `size` 定义。

> [!NOTE]
> 对全局默认 dtype `torch.float32`，该函数返回张量的 dtype 为 `torch.int64`

**参数：**

- **low** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – Lowest integer to be drawn from the distribution. Default: 0.
- **high** ([*int*](https://docs.python.org/3/library/functions.html#int)) – One above the highest integer to be drawn from the distribution.
- **size** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – a tuple defining the shape of the output tensor.


### torch.randn

```python
torch.randn(*size, *, 
    out=None, 
    dtype=None, 
    layout=torch.strided, 
    device=None, 
    requires_grad=False, 
    pin_memory=False) → Tensor
```

返回一个由均值为0、方差为 1 的正态分布中随机数填充的张量。

$$out_i ∼ N(0,1)$$

张量的形状由参数 `size` 定义。

**参数：**

- **size** (`int`...)

整数序列，定义输出张量的形状。可以是多个可变参数，也可以是 list 或 tuple。

- **generator** ([torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), optional)

用于抽样的伪随机数生成器。

- **out** (`Tensor`, optional)

输出张量。

- **dtype** (`torch.dtype`, optional)

返回张量的期望数据类型。如果为 `None`，则使用全局默认类型，参考 `torch.set_default_tensor_type()`。

```python
>>> torch.randn(4)
tensor([-0.6837, -0.0592,  1.2451, -0.8639])
>>> torch.randn(2, 3)
tensor([[ 0.6635, -1.0228,  0.0674],
        [ 1.4007,  1.6177, -0.7507]])
```

### 原地随机采样

## 局部禁用梯度计算

上下文管理器 `torch.no_grad()`, `torch.enable_grad()` 和 `torch.set_grad_enabled()` 对局部禁用或启用梯度计算非常有用。

no_grad

Context-manager that disabled gradient calculation.

enable_grad

Context-manager that enables gradient calculation.

set_grad_enabled

Context-manager that sets gradient calculation to on or off.

is_grad_enabled

Returns True if grad mode is currently enabled.

inference_mode

Context-manager that enables or disables inference mode

is_inference_mode_enabled

Returns True if inference mode is currently enabled.

## 数学运算

对任何具有相同 shape 的张量，常见的标准算法运算符（+, -, *, / 和 **） 都可以升级为逐元素运算。

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
```



### Pointwise Ops

#### abs

> 2024年10月22日 ⭐

```python
torch.abs(input, *, out=None) → Tensor
```

计算 `input` 中每个元素的绝对值。
$$
\text{out}_i=|\text{input}_i|
$$
参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入 tensor

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出 tensor

```python
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([ 1,  2,  3])
```

#### absolute

> 2024年10月22日 ⭐

[torch.abs()](#abs) 的别名。



acos

Computes the inverse cosine of each element in input.

arccos

Alias for torch.acos().

acosh

Returns a new tensor with the inverse hyperbolic cosine of the elements of input.

arccosh

Alias for torch.acosh().

#### add

> 2024年10月22日 ⭐

```python
torch.add(input, other, *, alpha=1, out=None) → Tensor
```

计算：
$$
\text{out}_i=\text{input}_i+\text{alpha}\times\text{other}_i
$$

支持常见 shape 广播、类型提升以及 integer, float 和 complex 输入。

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入 tensor
- **other** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or* *Number*) – 加到 `input` 的 tensor 或数字

**关键字参数：**

- **alpha** (*Number*) – `other` 的系数
- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出 tensor

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
>>> torch.add(a, 20)
tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

>>> b = torch.randn(4)
>>> b
tensor([-0.9732, -0.3497,  0.6245,  0.4022])
>>> c = torch.randn(4, 1)
>>> c
tensor([[ 0.3743],
        [-1.7724],
        [-0.5811],
        [-0.8017]])
>>> torch.add(b, c, alpha=10)
tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
        [-18.6971, -18.0736, -17.0994, -17.3216],
        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
```



addcdiv

Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input.

addcmul

Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the scalar value and add it to input.

angle

Computes the element-wise angle (in radians) of the given input tensor.

asin

Returns a new tensor with the arcsine of the elements of input.

arcsin

Alias for torch.asin().

asinh

Returns a new tensor with the inverse hyperbolic sine of the elements of input.

arcsinh

Alias for torch.asinh().

atan

Returns a new tensor with the arctangent of the elements of input.

arctan

Alias for torch.atan().

atanh

Returns a new tensor with the inverse hyperbolic tangent of the elements of input.

arctanh

Alias for torch.atanh().

atan2

Element-wise arctangent of \text{input}_{i} / \text{other}_{i}input 
i
​
 /other 
i
​
  with consideration of the quadrant.

arctan2

Alias for torch.atan2().

bitwise_not

Computes the bitwise NOT of the given input tensor.

bitwise_and

Computes the bitwise AND of input and other.

bitwise_or

Computes the bitwise OR of input and other.

bitwise_xor

Computes the bitwise XOR of input and other.

bitwise_left_shift

Computes the left arithmetic shift of input by other bits.

bitwise_right_shift

Computes the right arithmetic shift of input by other bits.

ceil

Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.

clamp

Clamps all elements in input into the range [ min, max ].

clip

Alias for torch.clamp().

conj_physical

Computes the element-wise conjugate of the given input tensor.

copysign

Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.

cos

Returns a new tensor with the cosine of the elements of input.

cosh

Returns a new tensor with the hyperbolic cosine of the elements of input.

deg2rad

Returns a new tensor with each of the elements of input converted from angles in degrees to radians.

div

Divides each element of the input input by the corresponding element of other.

divide

Alias for torch.div().

digamma

Alias for torch.special.digamma().

erf

Alias for torch.special.erf().

erfc

Alias for torch.special.erfc().

erfinv

Alias for torch.special.erfinv().

#### exp

> 2024年10月22日 ⭐

```python
torch.exp(input, *, out=None) → Tensor
```

计算 `input` 的自然指数，得到一个新张量。
$$
y_i=e^{x_i}
$$

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入 tensor

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出 tensor

```python
>>> torch.exp(torch.tensor([0, math.log(2.)]))
tensor([ 1.,  2.])
```





exp2

Alias for torch.special.exp2().

expm1

Alias for torch.special.expm1().

fake_quantize_per_channel_affine

Returns a new tensor with the data in input fake quantized per channel using scale, zero_point, quant_min and quant_max, across the channel specified by axis.

fake_quantize_per_tensor_affine

Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min and quant_max.

fix

Alias for torch.trunc()

float_power

Raises input to the power of exponent, elementwise, in double precision.

floor

Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.

floor_divide

fmod

Applies C++'s std::fmod entrywise.

frac

Computes the fractional portion of each element in input.

frexp

Decomposes input into mantissa and exponent tensors such that \text{input} = \text{mantissa} \times 2^{\text{exponent}}input=mantissa×2 
exponent
 .

gradient

Estimates the gradient of a function g : \mathbb{R}^n \rightarrow \mathbb{R}g:R 
n
 →R in one or more dimensions using the second-order accurate central differences method.

imag

Returns a new tensor containing imaginary values of the self tensor.

ldexp

Multiplies input by 2**:attr:other.

lerp

Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.

lgamma

Computes the natural logarithm of the absolute value of the gamma function on input.

log

Returns a new tensor with the natural logarithm of the elements of input.

log10

Returns a new tensor with the logarithm to the base 10 of the elements of input.

log1p

Returns a new tensor with the natural logarithm of (1 + input).

log2

Returns a new tensor with the logarithm to the base 2 of the elements of input.

logaddexp

Logarithm of the sum of exponentiations of the inputs.

logaddexp2

Logarithm of the sum of exponentiations of the inputs in base-2.

logical_and

Computes the element-wise logical AND of the given input tensors.

logical_not

Computes the element-wise logical NOT of the given input tensor.

logical_or

Computes the element-wise logical OR of the given input tensors.

logical_xor

Computes the element-wise logical XOR of the given input tensors.

logit

Alias for torch.special.logit().

hypot

Given the legs of a right triangle, return its hypotenuse.

i0

Alias for torch.special.i0().

igamma

Alias for torch.special.gammainc().

igammac

Alias for torch.special.gammaincc().

mul

Multiplies input by other.

multiply

Alias for torch.mul().

mvlgamma

Alias for torch.special.multigammaln().

nan_to_num

Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.

neg

Returns a new tensor with the negative of the elements of input.

negative

Alias for torch.neg()

nextafter

Return the next floating-point value after input towards other, elementwise.

polygamma

Alias for torch.special.polygamma().

positive

Returns input.

pow

Takes the power of each element in input with exponent and returns a tensor with the result.

quantized_batch_norm

Applies batch normalization on a 4D (NCHW) quantized tensor.

quantized_max_pool1d

Applies a 1D max pooling over an input quantized tensor composed of several input planes.

quantized_max_pool2d

Applies a 2D max pooling over an input quantized tensor composed of several input planes.

rad2deg

Returns a new tensor with each of the elements of input converted from angles in radians to degrees.

real

Returns a new tensor containing real values of the self tensor.

reciprocal

Returns a new tensor with the reciprocal of the elements of input

remainder

Computes Python's modulus operation entrywise.

round

Rounds elements of input to the nearest integer.

rsqrt

Returns a new tensor with the reciprocal of the square-root of each of the elements of input.

sigmoid

Alias for torch.special.expit().

sign

Returns a new tensor with the signs of the elements of input.

sgn

This function is an extension of torch.sign() to complex tensors.

signbit

Tests if each element of input has its sign bit set or not.

sin

Returns a new tensor with the sine of the elements of input.

sinc

Alias for torch.special.sinc().

sinh

Returns a new tensor with the hyperbolic sine of the elements of input.

|[sqrt](#torchsqrt)|逐个计算输入张量每个元素的平方根|

square

Returns a new tensor with the square of the elements of input.

sub

Subtracts other, scaled by alpha, from input.

subtract

Alias for torch.sub().

tan

Returns a new tensor with the tangent of the elements of input.

tanh

Returns a new tensor with the hyperbolic tangent of the elements of input.

true_divide

Alias for torch.div() with rounding_mode=None.

trunc

Returns a new tensor with the truncated integer values of the elements of input.

xlogy

Alias for torch.special.xlogy().


### 降维操作

argmax

Returns the indices of the maximum value of all elements in the input tensor.

argmin

Returns the indices of the minimum value(s) of the flattened tensor or along a dimension

amax

Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.

#### torch.amin

```python
torch.amin(
    input, 
    dim, 
    keepdim=False, 
    *, 
    out=None) → Tensor
```

返回 `input` 张量 `dim` 维度每个切片的最小值。

`max`/`min` 和 `amax`/`amin` 的差别：

- `amax`/`amin` 支持多维约简；
- `amax`/`amin` 不返回索引；
- `amax`/`amin` 在相等值之间均匀分布梯度，而 `max`/`min` 只将张量传播到原张量的单个索引。

如果 `keepdim=True`，则输出张量与 `input` 张量除了 `dim` 大小为 1，其它维度大小相同。否则，`dim` 被压缩（`torch.squeeze()`），输出张量维度减 1.

**参数：**

- **input** (`Tensor`) – 输入张量
- **dim** (`int` or tuple of ints) – 约简维度
- **keepdim** (`bool`) – 是否保留约简的维度

**关键字参数：**

- **out** (`Tensor`, optional) – 输出张量

**示例：**

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.6451, -0.4866,  0.2987, -1.3312],
        [-0.5744,  1.2980,  1.8397, -0.2713],
        [ 0.9128,  0.9214, -1.7268, -0.2995],
        [ 0.9023,  0.4853,  0.9075, -1.6165]])
>>> torch.amin(a, 1)
tensor([-1.3312, -0.5744, -1.7268, -1.6165])
```

aminmax

Computes the minimum and maximum values of the input tensor.

all

Tests if all elements in input evaluate to True.

any

Tests if any element in input evaluates to True.

max

Returns the maximum value of all elements in the input tensor.

min

Returns the minimum value of all elements in the input tensor.

dist

Returns the p-norm of (input - other)

logsumexp

Returns the log of summed exponentials of each row of the input tensor in the given dimension dim.

#### torch.mean

> 2024年10月23日⭐

```python
torch.mean(input, *, dtype=None) → Tensor
```

返回 `input` 张量所有元素的均值。`input` 张量必须诶 float 或 complex 类型。

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量，float 或 complex dtype

关键字参数：

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量的类型。如果指定类型，在操作前将 input 张量转换为 `dtype`。这对防止数据类型溢出很有用。默认 `None`

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.2294, -0.5481,  1.3288]])
>>> torch.mean(a)
tensor(0.3367)
```

```python
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None) → Tensor
```

返回 `input` 张量在 `dim` 维度的均值。如果 `dim` 为列表，则对所有这些维度降维。

如果 `keepdim` 为 `True`，则输出张量与 `input` 张量除了 `dim` 维度为 1，其它维度 size 相同。否则，`dim` 被压缩（参考 `torch.squeeze()`），使得输出张量维度数降低 `len(dim)`。

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量
- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* *ints,* *optional*) – 要降维的维度。`None` 表示对所有维度降维。
- **keepdim** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – 是否保留 `dim` 维度

关键字参数：

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量的类型。如果指定类型，在操作前将 input 张量转换为 `dtype`。这对防止数据类型溢出很有用。默认 `None`
- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
        [-0.9644,  1.0131, -0.6549, -1.4279],
        [-0.2951, -1.3350, -0.7694,  0.5600],
        [ 1.0842, -0.9580,  0.3623,  0.2343]])
>>> torch.mean(a, 1) # 对 row 求均值
tensor([-0.0163, -0.5085, -0.4599,  0.1807])
>>> torch.mean(a, 1, True)
tensor([[-0.0163],
        [-0.5085],
        [-0.4599],
        [ 0.1807]])
```






nanmean

Computes the mean of all non-NaN elements along the specified dimensions.

median

Returns the median of the values in input.

nanmedian

Returns the median of the values in input, ignoring NaN values.

mode

Returns a namedtuple (values, indices) where values is the mode value of each row of the input tensor in the given dimension dim, i.e. a value which appears most often in that row, and indices is the index location of each mode value found.

#### torch.norm

> 2024年10月23日⭐

```python
torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)
```

返回指定张量的矩阵范数或向量范数。

> [!WARNING]
>
> `torch.norm` 已弃用，在未来可能会从 PyTorch 中删除。其文档和行为可能不正确，且不再主动维护。
>
> 使用 `torch.linalg.vector_norm()` 计算向量范数，使用 `torch.linalg.matrix_norm()` 计算矩阵范数。函数 `torch.linalg.norm()` 与 `torch.norm()` 功能类似。



nansum

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

prod

Returns the product of all elements in the input tensor.

quantile

Computes the q-th quantiles of each row of the input tensor along the dimension dim.

nanquantile

This is a variant of torch.quantile() that "ignores" NaN values, computing the quantiles q as if NaN values in input did not exist.

std

If unbiased is True, Bessel's correction will be used.

std_mean

If unbiased is True, Bessel's correction will be used to calculate the standard deviation.

#### torch.sum

> 2024年10月23日⭐

```python
torch.sum(input, *, dtype=None) → Tensor
```

返回 `input` 张量所有元素和。

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量。

**关键字参数：**

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量的类型。如果指定类型，在操作前将 input 张量转换为 `dtype`。这对防止数据类型溢出很有用。默认 `None`

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.1133, -0.9567,  0.2958]])
>>> torch.sum(a)
tensor(-0.5475)
```

```python
torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
```

返回 `input` 张量在 `dim` 维度的加和。如果 `dim` 为列表，则对所有这些维度降维。

如果 `keepdim` 为 `True`，则输出张量与 `input` 张量除了 `dim` 维度为 1，其它维度 size 相同。否则，`dim` 倍压缩（参考 `torch.squeeze()`），使得输出张量维度数降低 `len(dim)`。

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量
- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* *ints,* *optional*) – 要降维的维度。`None` 表示对所有维度降维。
- **keepdim** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – 是否保留 `dim` 维度

关键字参数：

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量的类型。如果指定类型，在操作前将 input 张量转换为 `dtype`。这对防止数据类型溢出很有用。默认 `None`

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, 1) # 4x4 变为 4x0
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
>>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
>>> torch.sum(b, (2, 1))
tensor([  435.,  1335.,  2235.,  3135.])
```





unique

Returns the unique elements of the input tensor.

unique_consecutive

Eliminates all but the first element from every consecutive group of equivalent elements.

|[var](#torchvar)|计算方差|


var_mean

If unbiased is True, Bessel's correction will be used to calculate the variance.

count_nonzero

Counts the number of non-zero values in the tensor input along the given dim.

### 比较

allclose

This function checks if all input and other satisfy the condition:

argsort

Returns the indices that sort a tensor along a given dimension in ascending order by value.

#### torch.eq

```python
torch.eq(input, other, *, out=None) → Tensor
```

计算每个元素是否相等。

第二个参数可以是数字或张量，第二个张量的 shape 可以广播到与第一个参数匹配。

**参数：**

- **input** (`Tensor`) – 要比较的张量
- **other** (`Tensor` or `float`) – 要比较的张量或数值

**关键字参数：**

- **out** (`Tensor`, optional) – 输出张量

**返回：**

boolean 张量，`input` 与 `other` 相等时为 `True`.

**示例：**

```python
>>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
tensor([[ True, False],
        [False, True]])
```

equal

True if two tensors have the same size and elements, False otherwise.

ge

Computes 
input
≥
other
input≥other element-wise.

greater_equal

Alias for torch.ge().

gt

Computes 
input
>
other
input>other element-wise.

greater

Alias for torch.gt().

isclose

Returns a new tensor with boolean elements representing if each element of input is "close" to the corresponding element of other.

isfinite

Returns a new tensor with boolean elements representing if each element is finite or not.

#### torch.isin

```python
torch.isin(
    elements, 
    test_elements, 
    *, 
    assume_unique=False, 
    invert=False) → Tensor
```

测试 `elements` 的每个元素是否在 `test_elements` 中。返回一个与 `elements` shape 相同的 boolean 张量，当 `elements` 中对应元素在 `test_elements` 时为 True，否则为 False.

> **NOTE**
> `elements` 或 `test_elements` 其中一个可以为标量，但不能都是标量。

**参数:**

- **elements** (`Tensor` or Scalar) – 输入元素
- **test_elements** (`Tensor` or Scalar) – 用于测试输入元素
- **assume_unique** (`bool`, optional) – True 则假设 `elements` 和 `test_elements` 不含重复值，这样可以加速计算速度。默认：False
- **invert** (`bool`, optional) – True 则反转 boolean 张量，此时不在 `test_elements` 中的元素结果为 True。默认：False

**返回：**

- 一个 shape 与 `elements` 相同的布尔张量，当对应元素在 `test_elements` 中为 True，否则为 False。

**示例：**

```python
>>> torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))
tensor([[False,  True],
        [ True, False]])
```

isinf

Tests if each element of input is infinite (positive or negative infinity) or not.

isposinf

Tests if each element of input is positive infinity or not.

isneginf

Tests if each element of input is negative infinity or not.

isnan

Returns a new tensor with boolean elements representing if each element of input is NaN or not.

isreal

Returns a new tensor with boolean elements representing if each element of input is real-valued or not.

kthvalue

Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim.

le

Computes 
input
≤
other
input≤other element-wise.

less_equal

Alias for torch.le().

lt

Computes 
input
<
other
input<other element-wise.

less

Alias for torch.lt().

maximum

Computes the element-wise maximum of input and other.

minimum

Computes the element-wise minimum of input and other.

fmax

Computes the element-wise maximum of input and other.

fmin

Computes the element-wise minimum of input and other.

ne

Computes 
input
≠
other
input

=other element-wise.

not_equal

Alias for torch.ne().

sort

Sorts the elements of the input tensor along a given dimension in ascending order by value.

topk

Returns the k largest elements of the given input tensor along a given dimension.

msort

Sorts the elements of the input tensor along its first dimension in ascending order by value.

### Spectral Ops

### 其它操作

atleast_1d

Returns a 1-dimensional view of each input tensor with zero dimensions.

atleast_2d

Returns a 2-dimensional view of each input tensor with zero dimensions.

atleast_3d

Returns a 3-dimensional view of each input tensor with zero dimensions.

bincount

Count the frequency of each value in an array of non-negative ints.

block_diag

Create a block diagonal matrix from provided tensors.

broadcast_tensors

Broadcasts the given tensors according to Broadcasting semantics.

broadcast_to

Broadcasts input to the shape shape.

broadcast_shapes

Similar to broadcast_tensors() but for shapes.

bucketize

Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by boundaries.

cartesian_prod

Do cartesian product of the given sequence of tensors.

cdist

Computes batched the p-norm distance between each pair of the two collections of row vectors.

#### torch.clone

```python
torch.clone(input, *, 
    memory_format=torch.preserve_format) → Tensor
```

返回 `input` 的副本。

> **NOTE**
> 该操作是可微的，所以梯度会从这个操作的结果流回 `input`。要创建于 `input` 没有 autograd 关系的张量，可参考 `detach()`。




combinations

Compute combinations of length 
�
r of the given tensor.

corrcoef

Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.

cov

Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.

cross

Returns the cross product of vectors in dimension dim of input and other.

cummax

Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim.

cummin

Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim.

cumprod

Returns the cumulative product of elements of input in the dimension dim.

#### torch.cumsum

> 2024年10月23日⭐

```python
torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor
```

返回 `input` 张量在 `dim` 维度的累计加和。

例如，如果 `input` 是 size 为 N 的向量，那么结果也是 size 为 N 的向量，元素为：
$$
y_i=x_1+x_2+x_3+\cdots+x_i
$$
参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 输入张量
- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – 执行操作的维度

关键字参数：

- **dtype** ([`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), optional) – 返回张量的类型。如果指定类型，在操作前将 input 张量转换为 `dtype`。这对防止数据类型溢出很有用。默认 `None`
- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> a = torch.randint(1, 20, (10,))
>>> a
tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
>>> torch.cumsum(a, dim=0)
tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
```



diag

If input is a vector (1-D tensor), then returns a 2-D square tensor

diag_embed

Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input.

diagflat

If input is a vector (1-D tensor), then returns a 2-D square tensor

diagonal

Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.

diff

Computes the n-th forward difference along the given dimension.

einsum

Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.

flatten

Flattens input by reshaping it into a one-dimensional tensor.

#### torch.flip

```python
torch.flip(input, dims) → Tensor
```

将张量指定维度的数据翻转。如对序列数据，生成 reverse 序列数据。

> **NOTE**
> `torch.flip` 复制 `input` 数据，而 NumPy 的 `np.flip` 返回一个 view。因为复制张量更费时，所以 `torch.flip` 预计比 `np.flip` 慢。

**参数：**

- **input** (`Tensor`) – 输入张量。
- **dims** (a list or `tuple`) – 翻转的维度。

**示例：**

```python
>>> x = torch.arange(8).view(2, 2, 2)
>>> x
tensor([[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]]])
>>> torch.flip(x, [0, 1]) # 同时翻转两个维度
tensor([[[ 6,  7],
         [ 4,  5]],

        [[ 2,  3],
         [ 0,  1]]])
```

fliplr

Flip tensor in the left/right direction, returning a new tensor.

flipud

Flip tensor in the up/down direction, returning a new tensor.

kron

Computes the Kronecker product, denoted by 
⊗
⊗, of input and other.

rot90

Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.

gcd

Computes the element-wise greatest common divisor (GCD) of input and other.

histc

Computes the histogram of a tensor.

histogram

Computes a histogram of the values in a tensor.

histogramdd

Computes a multi-dimensional histogram of the values in a tensor.

meshgrid

Creates grids of coordinates specified by the 1D inputs in attr:tensors.

lcm

Computes the element-wise least common multiple (LCM) of input and other.

logcumsumexp

Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.

ravel

Return a contiguous flattened tensor.

renorm

Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm

repeat_interleave

Repeat elements of a tensor.

roll

Roll the tensor input along the given dimension(s).

searchsorted

Find the indices from the innermost dimension of sorted_sequence such that, if the corresponding values in values were inserted before the indices, when sorted, the order of the corresponding innermost dimension within sorted_sequence would be preserved.

tensordot

Returns a contraction of a and b over multiple dimensions.

trace

Returns the sum of the elements of the diagonal of the input 2-D matrix.

tril

Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

tril_indices

Returns the indices of the lower triangular part of a row-by- col matrix in a 2-by-N Tensor, where the first row contains row coordinates of all indices and the second row contains column coordinates.

triu

Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

triu_indices

Returns the indices of the upper triangular part of a row by col matrix in a 2-by-N Tensor, where the first row contains row coordinates of all indices and the second row contains column coordinates.

unflatten

Expands a dimension of the input tensor over multiple dimensions.

vander

Generates a Vandermonde matrix.

view_as_real

Returns a view of input as a real tensor.

view_as_complex

Returns a view of input as a complex tensor.

resolve_conj

Returns a new tensor with materialized conjugation if input's conjugate bit is set to True, else returns input.

resolve_neg

Returns a new tensor with materialized negation if input's negative bit is set to True, else returns input.

### BLAS 和 LAPACK 操作



#### torch.bmm

```python
torch.bmm(input, mat2, *, out=None) → Tensor
```

执行 `input` 和 `mat2` 矩阵的批处理矩阵乘法。

`input` 和 `mat2` 是包含相同数目矩阵的 3D 张量。

如果 `input` 为 $(b\times n\times m)$ 张量，`mat2` 是 $(b\times m\times p)$ 张量，则 `out` 为 $(b\times n\times p)$ 张量。

$$out_i=input_i @ mat2_i$$

> **NOTE**
> 该函数不支持广播，对矩阵广播乘法，使用 `torch.matmul()`。

**参数**

- **input** (`Tensor`)：第一批矩阵
- **mat2** (`Tensor`)：第二批矩阵 

```python
>>> input = torch.randn(10, 3, 4)
>>> mat2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(input, mat2)
>>> res.size()
torch.Size([10, 3, 5])
```

| [`addbmm`](https://pytorch.org/docs/stable/generated/torch.addbmm.html#torch.addbmm) | Performs a batch matrix-matrix product of matrices stored in `batch1` and `batch2`, with a reduced add step (all matrix multiplications get accumulated along the first dimension). |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`addmm`](https://pytorch.org/docs/stable/generated/torch.addmm.html#torch.addmm) | Performs a matrix multiplication of the matrices `mat1` and `mat2`. |
| [`addmv`](https://pytorch.org/docs/stable/generated/torch.addmv.html#torch.addmv) | Performs a matrix-vector product of the matrix `mat` and the vector `vec`. |
| [`addr`](https://pytorch.org/docs/stable/generated/torch.addr.html#torch.addr) | Performs the outer-product of vectors `vec1` and `vec2` and adds it to the matrix `input`. |
| [`baddbmm`](https://pytorch.org/docs/stable/generated/torch.baddbmm.html#torch.baddbmm) | Performs a batch matrix-matrix product of matrices in `batch1` and `batch2`. |
| [`bmm`](https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm) | Performs a batch matrix-matrix product of matrices stored in `input` and `mat2`. |
| [`chain_matmul`](https://pytorch.org/docs/stable/generated/torch.chain_matmul.html#torch.chain_matmul) | Returns the matrix product of the N*N* 2-D tensors.          |
| [`cholesky`](https://pytorch.org/docs/stable/generated/torch.cholesky.html#torch.cholesky) | Computes the Cholesky decomposition of a symmetric positive-definite matrix A*A* or for batches of symmetric positive-definite matrices. |
| [`cholesky_inverse`](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch.cholesky_inverse) | Computes the inverse of a complex Hermitian or real symmetric positive-definite matrix given its Cholesky decomposition. |
| [`cholesky_solve`](https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html#torch.cholesky_solve) | Computes the solution of a system of linear equations with complex Hermitian or real symmetric positive-definite lhs given its Cholesky decomposition. |
| [`dot`](https://pytorch.org/docs/stable/generated/torch.dot.html#torch.dot) | Computes the dot product of two 1D tensors.                  |
| [`geqrf`](https://pytorch.org/docs/stable/generated/torch.geqrf.html#torch.geqrf) | This is a low-level function for calling LAPACK's geqrf directly. |
| [`ger`](https://pytorch.org/docs/stable/generated/torch.ger.html#torch.ger) | Alias of [`torch.outer()`](https://pytorch.org/docs/stable/generated/torch.outer.html#torch.outer). |
| [`inner`](https://pytorch.org/docs/stable/generated/torch.inner.html#torch.inner) | Computes the dot product for 1D tensors.                     |
| [`inverse`](https://pytorch.org/docs/stable/generated/torch.inverse.html#torch.inverse) | Alias for [`torch.linalg.inv()`](https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv) |
| [`det`](https://pytorch.org/docs/stable/generated/torch.det.html#torch.det) | Alias for [`torch.linalg.det()`](https://pytorch.org/docs/stable/generated/torch.linalg.det.html#torch.linalg.det) |
| [`logdet`](https://pytorch.org/docs/stable/generated/torch.logdet.html#torch.logdet) | Calculates log determinant of a square matrix or batches of square matrices. |
| [`slogdet`](https://pytorch.org/docs/stable/generated/torch.slogdet.html#torch.slogdet) | Alias for [`torch.linalg.slogdet()`](https://pytorch.org/docs/stable/generated/torch.linalg.slogdet.html#torch.linalg.slogdet) |
| [`lu`](https://pytorch.org/docs/stable/generated/torch.lu.html#torch.lu) | Computes the LU factorization of a matrix or batches of matrices `A`. |
| [`lu_solve`](https://pytorch.org/docs/stable/generated/torch.lu_solve.html#torch.lu_solve) | Returns the LU solve of the linear system Ax=b*A**x*=*b* using the partially pivoted LU factorization of A from [`lu_factor()`](https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor). |
| [`lu_unpack`](https://pytorch.org/docs/stable/generated/torch.lu_unpack.html#torch.lu_unpack) | Unpacks the LU decomposition returned by [`lu_factor()`](https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor) into the P, L, U matrices. |
| [`matrix_power`](https://pytorch.org/docs/stable/generated/torch.matrix_power.html#torch.matrix_power) | Alias for [`torch.linalg.matrix_power()`](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_power.html#torch.linalg.matrix_power) |
| [`matrix_exp`](https://pytorch.org/docs/stable/generated/torch.matrix_exp.html#torch.matrix_exp) | Alias for [`torch.linalg.matrix_exp()`](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html#torch.linalg.matrix_exp). |
| [`orgqr`](https://pytorch.org/docs/stable/generated/torch.orgqr.html#torch.orgqr) | Alias for [`torch.linalg.householder_product()`](https://pytorch.org/docs/stable/generated/torch.linalg.householder_product.html#torch.linalg.householder_product). |
| [`ormqr`](https://pytorch.org/docs/stable/generated/torch.ormqr.html#torch.ormqr) | Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix. |
| [`outer`](https://pytorch.org/docs/stable/generated/torch.outer.html#torch.outer) | Outer product of `input` and `vec2`.                         |
| [`pinverse`](https://pytorch.org/docs/stable/generated/torch.pinverse.html#torch.pinverse) | Alias for [`torch.linalg.pinv()`](https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html#torch.linalg.pinv) |
| [`qr`](https://pytorch.org/docs/stable/generated/torch.qr.html#torch.qr) | Computes the QR decomposition of a matrix or a batch of matrices `input`, and returns a namedtuple (Q, R) of tensors such that input=QRinput=*QR* with Q*Q* being an orthogonal matrix or batch of orthogonal matrices and R*R* being an upper triangular matrix or batch of upper triangular matrices. |
| [`svd`](https://pytorch.org/docs/stable/generated/torch.svd.html#torch.svd) | Computes the singular value decomposition of either a matrix or batch of matrices `input`. |
| [`svd_lowrank`](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html#torch.svd_lowrank) | Return the singular value decomposition `(U, S, V)` of a matrix, batches of matrices, or a sparse matrix A*A* such that A≈Udiag⁡(S)VH*A*≈*U*diag(*S*)*V*H. |
| [`pca_lowrank`](https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html#torch.pca_lowrank) | Performs linear Principal Component Analysis (PCA) on a low-rank matrix, batches of such matrices, or sparse matrix. |
| [`lobpcg`](https://pytorch.org/docs/stable/generated/torch.lobpcg.html#torch.lobpcg) | Find the k largest (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric positive definite generalized eigenvalue problem using matrix-free LOBPCG methods. |
| [`trapz`](https://pytorch.org/docs/stable/generated/torch.trapz.html#torch.trapz) | Alias for [`torch.trapezoid()`](https://pytorch.org/docs/stable/generated/torch.trapezoid.html#torch.trapezoid). |
| [`trapezoid`](https://pytorch.org/docs/stable/generated/torch.trapezoid.html#torch.trapezoid) | Computes the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) along `dim`. |
| [`cumulative_trapezoid`](https://pytorch.org/docs/stable/generated/torch.cumulative_trapezoid.html#torch.cumulative_trapezoid) | Cumulatively computes the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) along `dim`. |
| [`triangular_solve`](https://pytorch.org/docs/stable/generated/torch.triangular_solve.html#torch.triangular_solve) | Solves a system of equations with a square upper or lower triangular invertible matrix A*A* and multiple right-hand sides b*b*. |
| [`vdot`](https://pytorch.org/docs/stable/generated/torch.vdot.html#torch.vdot) | Computes the dot product of two 1D vectors along a dimension. |

#### torch.matmul

> 2024-10-24⭐

```python
torch.matmul(input, other, *, out=None) → Tensor
```

两个张量的矩阵乘积。

具体行为取决于两个张量的维度：

- 如果两个张量都是 1D，返回点积（标量）
- 如果两个张量都是 2D，返回 matrix-matrix 乘积
- 如果参数 1 为 1D，参数 2 为 2D，则在参数 1 前面加上 1 维以进行矩阵乘法
- 如果参数 1 为 2D，参数 2 为 1D，则返回 matrix-vector 乘积
- 如果两个参数都不低于 1D，且其中一个 >2D，则执行 batch 矩阵乘法。
  - 如果参数 1 是 1D，则在前面添加 1 以执行 batch 矩阵乘法
  - 如果参数 2 是 1D，则在前面添加 1 以执行 batch 矩阵乘法
  - 对 non-matrix 维度（即 batch）进行广播（因此必须可广播）。例如，如果 `input` 是 $(j\times 1\times n\times n)$ 张量，`other` 是 $(k\times n\times n)$ 张量，那么 `out` 是 $(j\times k\times n\times n)$ 张量。在确定输入是否可广播时仅查看 batch 维度，不看矩阵维度。例如，如果 `input` 是 $(j\times 1\times n\times m)$ 张量，`other` 是 $(k\times m\times p)$ 张量，虽然最后两个维度不同，但是可广播的。`out` 为 $(j\times k\times n\times p)$ 张量。

该操作支持 sparse-layout 参数。其中 matrix-matrix 对稀疏参数的限制与 `torch.mm` 相同。

该操作支持 `TensorFloat32`。

在某些 ROCm 设备，当使用 float16 输入时，该模块使用摆脱那个的精度进行后向传播。

> [!NOTE]
> 此函数的 1D dot-product 版本不支持 `out` 参数。

**参数：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 乘法的第一个张量
- **other** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 乘法的第二个张量

**关键字参数：**

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> # vector x vector: dot-product
>>> tensor1 = torch.randn(3)
>>> tensor2 = torch.randn(3)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([])
>>> # matrix x vector: (3, 4)x(4, 1)=(3, 1)
>>> tensor1 = torch.randn(3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([3])
>>> # batched matrix x broadcasted vector: (10, 3, 4)x(4, 1)=(10, 3, 1)
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3])
>>> # batched matrix x batched matrix: (10, 3, 4)x(10, 4, 5)=(10, 3, 5)
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(10, 4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
>>> # batched matrix x broadcasted matrix: (10, 3, 4)x(4, 5)=(10, 3, 5)
>>> tensor1 = torch.randn(10, 3, 4)
>>> tensor2 = torch.randn(4, 5)
>>> torch.matmul(tensor1, tensor2).size()
torch.Size([10, 3, 5])
```

#### torch.mm

```python
torch.mm(input, mat2, *, out=None) → Tensor
```

计算矩阵 `input` 和 `mat2` 的矩阵乘法。

如果 `input` 是 $(n\times m)$ 张量，mat2 是 $(m\times p)$ 张量，则输出 `out` 为 $(n\times p)$ 张量。

> [!NOTE]
>
> 该函数不支持广播，对广播矩阵乘法，使用 `torch.matmul()`

支持 strided 和 sparse 2D 张量，autograd 支持 strided 输入。

该操作支持

#### torch.mv

> 2024年10月23日⭐

```python
torch.mv(input, vec, *, out=None) → Tensor
```

对矩阵 `input` 和向量 `vec` 执行 matrix-vector 乘积。

如果 `input` 为 $n\times m$ 张量，`vec` 是 size 为 $m$ 的 1D 张量，那么输出 `out` 是 size 为 $n$ 的 1D 张量。

> [!NOTE]
>
> 该操作不支持[广播](../tutorials/notes/broadcasting.md)。

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 要相乘的矩阵
- **vec** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 要相乘的向量

关键字参数：

- **out** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 输出张量

```python
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.mv(mat, vec)
tensor([ 1.0404, -0.6361])
```



## Utilities

### use_deterministic_algorithms

```python
torch.use_deterministic_algorithms(mode, *, warn_only=False)
```

设置 PyTorch 操作是否使用**确定性** (deterministic) 算法。确定性算法，指给定相同输入，在相同的软件和硬件上运行，总是生成相同的输出。启用该设置，则所有操作将使用确定性算法，如果只有不确定性算法可以，则在调用时抛出 `RuntimeError`。

> **NOTE:** 只用这个设置不足以使程序可重复。更多信息可参考 [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility)。

## Operator Tags

## 操作


### torch.mean

```python
torch.mean(input, *, dtype=None) → Tensor
```

计算输入张量 `input` 所有元素的均值。

参数：

- **input** (`Tensor`)：输入张量
- **dtype** (`torch.dtype`, optional)：返回张量的类型。指定后，在计算均值前将输入张量类型转换为 `dtype`。这对于防止数据类型溢出非常有用。

例如：

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.2294, -0.5481,  1.3288]])
>>> torch.mean(a)
tensor(0.3367)
```

```python
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None) → Tensor
```

计算张量 `input` 在指定维度 `dim` 每行的均值。如果 `dim` 是维度列表，则降维。

如果 `keepdim=True`，则输出张量除了 `dim` 为 1，其它维度大小与 `input` 相同。否则 `dim` 被压缩，使得输出张量的维度减 1。

参数：

- **input** (`Tensor`)：输入张量
- **dim** (`int` 或 tuple of ints)：待计算的维度
- **keepdim** (`bool`)：输出张量的 `dim` 维度是否保留
- **dtype** (`torch.dtype`, optional)：返回张量的类型。指定后，在计算均值前将输入张量类型转换为 `dtype`。这对于防止数据类型溢出非常有用。
- **out** (`Tensor`, optional)：输出张量

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
        [-0.9644,  1.0131, -0.6549, -1.4279],
        [-0.2951, -1.3350, -0.7694,  0.5600],
        [ 1.0842, -0.9580,  0.3623,  0.2343]])
>>> torch.mean(a, 1)
tensor([-0.0163, -0.5085, -0.4599,  0.1807])
>>> torch.mean(a, 1, True)
tensor([[-0.0163],
        [-0.5085],
        [-0.4599],
        [ 0.1807]])
```

### torch.reshape

> 2024年10月22日 ⭐

```python
torch.reshape(input, shape) → Tensor
```

返回一个与输入张量具有相同数据、元素个数和指定 shape 的张量。尽可能返回 `input` 的 view，否则返回副本。连续输入和 stride 兼容的输入可以不复制数据返回 reshape 视图，但是不应该依赖该行为，是 view 还是 copy 对 `reshape` 不确定。

其中一个维度可以指定为 -1，表示根据剩余维度和输入元素推断。

参数：

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the tensor to be reshaped
- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *of* [*int*](https://docs.python.org/3/library/functions.html#int)) – the new shape

```python
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
```

### torch.sqrt

```python
torch.sqrt(input, *, out=None) → Tensor
```

计算 `input` 张量所有元素的平方根。

```python
>>> a = torch.randn(4)
>>> a
tensor([-2.0755,  1.0226,  0.0831,  0.4806])
>>> torch.sqrt(a)
tensor([    nan,  1.0112,  0.2883,  0.6933])
```

### torch.sum

```python
torch.sum(input, *, dtype=None) → Tensor
```

计算张量 `input` 所有元素的加和。

参数：

- **input** (`Tensor`)：输入张量
- **dtype** (`torch.dtype`, optional)：返回张量的类型。指定后，在计算均值前将输入张量类型转换为 `dtype`。这对于防止数据类型溢出非常有用。

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.1133, -0.9567,  0.2958]])
>>> torch.sum(a)
tensor(-0.5475)
```

```python
torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
```

计算 `input` 张量指定维度 `dim` 元素加和。如果 `dim` 为维度列表，则对这些维度一起计算加和。

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, 1)
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
>>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
>>> torch.sum(b, (2, 1))
tensor([  435.,  1335.,  2235.,  3135.])
```

### torch.t

```python
torch.t(input) → Tensor
```

要求输入张量 `input` 维度 <= 2D，并转置维度 0 和 1.

0 维和 1 维张量按原样返回。对 2D 张量，等价于 `transpose(input, 0, 1)`。

例如：

```python
# 0 维张量，原样返回
>>> x = torch.randn(())
>>> x
tensor(0.1995)
>>> torch.t(x)
tensor(0.1995)
# 1 维张量，原样返回
>>> x = torch.randn(3)
>>> x
tensor([ 2.4320, -0.4608,  0.7702])
>>> torch.t(x)
tensor([ 2.4320, -0.4608,  0.7702])
# 2 维张量，进行转置
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.4875,  0.9158, -0.5872],
        [ 0.3938, -0.6929,  0.6932]])
>>> torch.t(x)
tensor([[ 0.4875,  0.3938],
        [ 0.9158, -0.6929],
        [-0.5872,  0.6932]])
```

### torch.var

```python
torch.var(input, dim, unbiased, keepdim=False, *, out=None) → Tensor
```

计算样本方差，如果 `unbiased=True`，应用 Bessel 校正。

参数：

- **input** (`Tensor`)：输入张量
- **dim** (`int` or tuple of ints, optional)：计算方差的维度，`None` 表示降维
- **unbiased** (`bool`)：是否使用 Bessel 校正
- **keepdim** (`bool`)：输出保留 `dim` 维度
- **out** (`Tensor`, optional)：输出张量

```python
torch.var(input, unbiased) → Tensor
```

计算张量 `input` 所有元素的方差。

如果 `unbiased=True`，使用 Bessel 校正。否则，不使用任何校正计算样本方差。

例如：

```python
>>> a = torch.tensor([[-0.8166, -1.3802, -0.3560]])
>>> torch.var(a, unbiased=False)
tensor(0.1754)
```

## 参考

- https://pytorch.org/docs/stable/torch.html
