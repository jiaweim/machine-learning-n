# torch

- [torch](#torch)
  - [Tensors](#tensors)
    - [创建操作](#创建操作)
    - [索引、切片、连接和突变](#索引切片连接和突变)
  - [随机抽样](#随机抽样)
  - [操作](#操作)
    - [torch.reshape](#torchreshape)
    - [torch.t](#torcht)
  - [参考](#参考)


## Tensors

|操作|说明|
|---|---|
|is_tensor|`obj` 是否为 PyTorch tensor|
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

### 创建操作

> **NOTE**：随机采用的创建操作在后面单独列出。

|操作|说明|
|---|---|
tensor

Constructs a tensor with no autograd history (also known as a "leaf tensor", see Autograd mechanics) by copying data.

sparse_coo_tensor

Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.

asarray

Converts obj to a tensor.

as_tensor

Converts data into a tensor, sharing data and preserving autograd history if possible.

as_strided

Create a view of an existing torch.Tensor input with specified size, stride and storage_offset.

from_numpy

Creates a Tensor from a numpy.ndarray.

from_dlpack

Converts a tensor from an external library into a torch.Tensor.

frombuffer

Creates a 1-dimensional Tensor from an object that implements the Python buffer protocol.

zeros

Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.

zeros_like

Returns a tensor filled with the scalar value 0, with the same size as input.

ones

Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.

ones_like

Returns a tensor filled with the scalar value 1, with the same size as input.

arange

Returns a 1-D tensor of size \left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil⌈ 
step
end−start
​
 ⌉ with values from the interval [start, end) taken with common difference step beginning from start.

range

Returns a 1-D tensor of size \left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1⌊ 
step
end−start
​
 ⌋+1 with values from start to end with step step.

linspace

Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.

logspace

Creates a one-dimensional tensor of size steps whose values are evenly spaced from {{\text{{base}}}}^{{\text{{start}}}}base 
start
  to {{\text{{base}}}}^{{\text{{end}}}}base 
end
 , inclusive, on a logarithmic scale with base base.

eye

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

empty

Returns a tensor filled with uninitialized data.

empty_like

Returns an uninitialized tensor with the same size as input.

empty_strided

Creates a tensor with the specified size and stride and filled with undefined data.

full

Creates a tensor of size size filled with fill_value.

full_like

Returns a tensor with the same size as input filled with fill_value.

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


### 索引、切片、连接和突变

|操作|说明|
|---|---|
|[reshape](#torchreshape)|返回指定形状的张量，与输入张量具有相同的数据和元素数|

## 随机抽样

## 操作

### torch.reshape

```python
torch.reshape(input, shape) → Tensor
```

返回一个与输入张量具有相同数据、元素和指定 shape 的张量。尽可能返回 `input` 的输入，否则返回副本。连续输入和 stride 兼容的输入可以不复制数据返回 reshape 视图，但是不应该依赖该行为，是 view 还是 copy 对 `reshape` 不确定。

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


## 参考

- https://pytorch.org/docs/stable/torch.html
