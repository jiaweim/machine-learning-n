# torch

- [torch](#torch)
  - [Tensors](#tensors)
    - [创建操作](#创建操作)
    - [索引、切片、连接和突变](#索引切片连接和突变)
      - [permute](#permute)
  - [Random sampling](#random-sampling)
    - [torch.manual\_seed](#torchmanual_seed)
  - [Math operations](#math-operations)
    - [Pointwise Ops](#pointwise-ops)
    - [Reduction Ops](#reduction-ops)
    - [Comparison Ops](#comparison-ops)
    - [Spectral Ops](#spectral-ops)
    - [Other Operations](#other-operations)
    - [BLAS and LAPACK Operations](#blas-and-lapack-operations)
      - [torch.bmm](#torchbmm)
      - [torch.mm](#torchmm)
  - [Utilities](#utilities)
    - [use\_deterministic\_algorithms](#use_deterministic_algorithms)
  - [Operator Tags](#operator-tags)
  - [操作](#操作)
    - [torch.adjoint](#torchadjoint)
    - [torch.argwhere](#torchargwhere)
    - [torch.mean](#torchmean)
    - [torch.reshape](#torchreshape)
    - [torch.sqrt](#torchsqrt)
    - [torch.sum](#torchsum)
    - [torch.t](#torcht)
    - [torch.unsqueeze](#torchunsqueeze)
    - [torch.var](#torchvar)
  - [参考](#参考)

Last updated: 2023-01-27, 13:39
***

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
|[adjoint](#torchadjoint)|将最后两个维度转换，然后返回共轭视图|
|[argwhere](#torchargwhere)|以张量形式返回 `input` 中非零元素索引|
|[reshape](#torchreshape)|返回指定形状的张量，与输入张量具有相同的数据和元素数|


cat

Concatenates the given sequence of seq tensors in the given dimension.

concat

Alias of torch.cat().

concatenate

Alias of torch.cat().

conj

Returns a view of input with a flipped conjugate bit.

chunk

Attempts to split a tensor into the specified number of chunks.

dsplit

Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.

column_stack

Creates a new tensor by horizontally stacking the tensors in tensors.

dstack

Stack tensors in sequence depthwise (along third axis).

gather

Gathers values along an axis specified by dim.

hsplit

Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.

hstack

Stack tensors in sequence horizontally (column wise).

index_add

See index_add_() for function description.

index_copy

See index_add_() for function description.

index_reduce

See index_reduce_() for function description.

index_select

Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.

masked_select

Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.

movedim

Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

moveaxis

Alias for torch.movedim().

narrow

Returns a new tensor that is a narrowed version of input tensor.

nonzero

#### permute

```python
torch.permute(input, dims) → Tensor
```

返回张量 `input` 的一个视图，其维度重新排列。

**参数：**

- **input** (`Tensor`)

输入张量。

- **dims** (tuple of python:int)

所需维度顺序。

```python
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```

reshape

Returns a tensor with the same data and number of elements as input, but with the specified shape.

row_stack

Alias of torch.vstack().

select

Slices the input tensor along the selected dimension at the given index.

scatter

Out-of-place version of torch.Tensor.scatter_()

diagonal_scatter

Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.

select_scatter

Embeds the values of the src tensor into input at the given index.

slice_scatter

Embeds the values of the src tensor into input at the given dimension.

scatter_add

Out-of-place version of torch.Tensor.scatter_add_()

scatter_reduce

Out-of-place version of torch.Tensor.scatter_reduce_()

split

Splits the tensor into chunks.

squeeze

Returns a tensor with all the dimensions of input of size 1 removed.

stack

Concatenates a sequence of tensors along a new dimension.

swapaxes

Alias for torch.transpose().

swapdims

Alias for torch.transpose().

t

Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

take

Returns a new tensor with the elements of input at the given indices.

take_along_dim

Selects values from input at the 1-dimensional indices from indices along the given dim.

tensor_split

Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.

tile

Constructs a tensor by repeating the elements of input.

transpose

Returns a tensor that is a transposed version of input.

unbind

Removes a tensor dimension.

|[unsqueeze](#torchunsqueeze)|在指定位置插入一个长度为 1 的维度|

vsplit

Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.

vstack

Stack tensors in sequence vertically (row wise).

where

Return a tensor of elements selected from either x or y, depending on condition.

## Random sampling

### torch.manual_seed

```python
torch.manual_seed(seed)
```

设置生成随机数的种子。返回 `torch.Generator` 对象。

参数：

- **seed** (`int`)

期望 seed。取值范围 [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]。负数按公式 0xffff_ffff_ffff_ffff + seed 重新映射到正数。

## Math operations

### Pointwise Ops

|操作|说明|
|---|---|
abs

Computes the absolute value of each element in input.

absolute

Alias for torch.abs()

acos

Computes the inverse cosine of each element in input.

arccos

Alias for torch.acos().

acosh

Returns a new tensor with the inverse hyperbolic cosine of the elements of input.

arccosh

Alias for torch.acosh().

add

Adds other, scaled by alpha, to input.

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

exp

Returns a new tensor with the exponential of the elements of the input tensor input.

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


### Reduction Ops

|操作|说明|
|---|---|
argmax

Returns the indices of the maximum value of all elements in the input tensor.

argmin

Returns the indices of the minimum value(s) of the flattened tensor or along a dimension

amax

Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.

amin

Returns the minimum value of each slice of the input tensor in the given dimension(s) dim.

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

|[mean](#torchmean)|返回输入张量所有元素的均值|


nanmean

Computes the mean of all non-NaN elements along the specified dimensions.

median

Returns the median of the values in input.

nanmedian

Returns the median of the values in input, ignoring NaN values.

mode

Returns a namedtuple (values, indices) where values is the mode value of each row of the input tensor in the given dimension dim, i.e. a value which appears most often in that row, and indices is the index location of each mode value found.

norm

Returns the matrix norm or vector norm of a given tensor.

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

sum

Returns the sum of all elements in the input tensor.

unique

Returns the unique elements of the input tensor.

unique_consecutive

Eliminates all but the first element from every consecutive group of equivalent elements.

|[var](#torchvar)|计算方差|


var_mean

If unbiased is True, Bessel's correction will be used to calculate the variance.

count_nonzero

Counts the number of non-zero values in the tensor input along the given dim.

### Comparison Ops

### Spectral Ops

### Other Operations

### BLAS and LAPACK Operations

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

#### torch.mm

```python
torch.mm(input, mat2, *, out=None) → Tensor
```

计算矩阵 `input` 和 `mat2` 的矩阵乘法。

如果 `input` 是 $(n\times m)$ 张量，mat2 是 $(m\times p)$ 张量，则输出 `out` 为 $(n\times p)$ 张量。

> **NOTE** 该函数不支持广播，对广播矩阵乘法，使用 `torch.matmul()`



## Utilities

### use_deterministic_algorithms

```python
torch.use_deterministic_algorithms(mode, *, warn_only=False)
```

设置 PyTorch 操作是否使用**确定性** (deterministic) 算法。确定性算法，指给定相同输入，在相同的软件和硬件上运行，总是生成相同的输出。启用该设置，则所有操作将使用确定性算法，如果只有不确定性算法可以，则在调用时抛出 `RuntimeError`。

> **NOTE:** 只用这个设置不足以使程序可重复。更多信息可参考 [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility)。

## Operator Tags

## 操作

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

```python
torch.reshape(input, shape) → Tensor
```

返回一个与输入张量具有相同数据、元素和指定 shape 的张量。尽可能返回 `input` 的输入，否则返回副本。连续输入和 stride 兼容的输入可以不复制数据返回 reshape 视图，但是不应该依赖该行为，是 view 还是 copy 对 `reshape` 不确定。

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

### torch.unsqueeze

```python
torch.unsqueeze(input, dim) → Tensor
```

在指定位置插入一个大小为 1 的维度，返回一个新的张量。

返回的张量与原张量共享底层数据。

`dim` 值的有效范围为 `[-input.dim() - 1, input.dim() + 1)`。负数 `dim` 对应维度为 `dim = dim + input.dim() + 1`。

例如：

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
