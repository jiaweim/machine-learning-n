# 张量

- [张量](#张量)
  - [1. 简介](#1-简介)
  - [2. 基础](#2-基础)
  - [3. shape](#3-shape)
  - [4. 索引](#4-索引)
    - [4.1 单轴索引](#41-单轴索引)
    - [4.2 多轴索引](#42-多轴索引)
  - [5. shape 操作](#5-shape-操作)
  - [6. dtype](#6-dtype)
  - [7. 广播](#7-广播)
  - [8. tf.convert_to_tensor](#8-tfconvert_to_tensor)
  - [9. 参差张量](#9-参差张量)
  - [10. 字符串张量](#10-字符串张量)
  - [11. 稀疏张量](#11-稀疏张量)
  - [12. 参考](#12-参考)

Last updated: 2022-09-22, 10:17
@author Jiawei Mao
*****

## 1. 简介

张量（Tensor）是具有统一类型（`dtype`）的多维数组，功能和 NumPy 的 `np.arrays` 类似。在 [tf.dtypes.DType](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType) 可以查看 TensorFlow 支持的所有数据类型。

所有的张量都是不可变的（immutable），因此不能修改张量内容，只能创建新的张量。

## 2. 基础

下面创建一些基本张量。

- 标量，即 0-阶（rank-0）张量。标量只包含一个值，没有轴（axes）

```python
import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4) # 默认为 int32 类型
print(rank_0_tensor)
```

```txt
tf.Tensor(4, shape=(), dtype=int32)
```

- 向量或 1-阶（rank-1）张量，类似列表，包含一个轴

```python
rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) # float 张量
print(rank_1_tensor)
```

```txt
tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
```

- 矩阵或 2-阶（rank-2）张量，包含2个轴

```python
# 使用 dtype 参数显式指定数据类型
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

```txt
tf.Tensor(
[[1. 2.]
 [3. 4.]
 [5. 6.]], shape=(3, 2), dtype=float16)
```

上面的三种张量的 shape 如下：

![](images/2021-12-21-10-13-58.png)

轴，即维度，张量可以有任意多个轴。例如，下面是一个三阶张量：

```python
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])
print(rank_3_tensor)
```

```txt
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
```

对高阶张量，可视化方法有多种。例如，对 shape 为 [3, 2, 5] 的 3 阶张量，可以按如下三种方式可视化：

![](images/2021-12-21-10-43-50.png)

用 `np.array` 或 `tensor.numpy` 方法可以将张量转换为 NumPy 数组：

```python
>>> np.array(rank_2_tensor)
array([[1., 2.],
       [3., 4.],
       [5., 6.]], dtype=float16)
```

```python
>>> rank_2_tensor.numpy()
array([[1., 2.],
       [3., 4.],
       [5., 6.]], dtype=float16)
```

张量通常为浮点型和整型，但也支持其它类型，包括：

- 复数
- 字符串

`tf.Tensor` 类要求张量是矩形的，即沿每个轴，每个元素大小相同。不过也有一些特殊的张量类型，可以处理不同 shape：

- [Ragged tensor](#9-参差张量)
- [Sparse tensor](#11-稀疏张量)

可以对张量进行基本的**数学运算**，例如：

```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])  # 也可以用 `tf.ones([2,2])`

print(tf.add(a, b), "\n") # 元素加
print(tf.multiply(a, b), "\n") # 元素乘
print(tf.matmul(a, b), "\n") # 矩阵乘
```

```txt
tf.Tensor(
[[2 3]
 [4 5]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[3 3]
 [7 7]], shape=(2, 2), dtype=int32) 
```

```python
print(a + b, "\n")  # 元素加
print(a * b, "\n")  # 元素乘
print(a @ b, "\n")  # 矩阵乘
```

输出同上。

张量还支持多种运算，如：

```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
# 查找最大值
print(tf.reduce_max(c))
# 查找最大值的索引
print(tf.math.argmax(c))
# 计算 softmax
print(tf.nn.softmax(c))
```

```txt
tf.Tensor(10.0, shape=(), dtype=float32)
tf.Tensor([1 0], shape=(2,), dtype=int64)
tf.Tensor(
[[2.6894143e-01 7.3105854e-01]
 [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)
```

> **NOTE:** 通常需要 `Tensor` 参数的 TF 函数，也支持能用 `tf.convert_to_tensor` 转换为 `Tensor` 的类型，示例如下。

```python
tf.convert_to_tensor([1, 2, 3])
```

```txt
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3])>
```

```python
tf.reduce_max([1, 2, 3])
```

```txt
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

```python
tf.reduce_max(np.array([1, 2, 3]))
```

```txt
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

## 3. shape

张量具有形状。首先介绍几个基本概念：

- **形状**（shape）：张量每个轴的长度（元素个数）。
- **秩**（rank）：张量轴的数目。如标量的秩是 0，向量的秩为 1，矩阵的秩为 2.
- **轴**（axis）或**维度**（dimension）：张量的特定维度。
- **尺寸**（size）：张量包含的元素个数，shape 向量的元素乘积。

![](images/2022-06-13-10-32-32.png)

> 不同 rank 张量的可视化表示

Tensor 和 [tf.TensorShape](https://www.tensorflow.org/api_docs/python/tf/TensorShape) 包含访问这些属性的方法。

```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

这是个 4 阶张量，shape `[3, 2, 4, 5]`，其属性如下：

![](images/2021-12-21-12-22-57.png)

```python
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```

```txt
Type of every element: <dtype: 'float32'>
Number of axes: 4
Shape of tensor: (3, 2, 4, 5)
Elements along axis 0 of tensor: 3
Elements along the last axis of tensor: 5
Total number of elements (3*2*4*5):  120
```

请注意，`Tensor.ndim` 和 `Tensor.shape` 属性返回的不是 `Tensor` 类型。如果需要 `Tensor` 类型，请使用 `tf.rank` 或 `tf.shape` 函数。这两者的区别很微妙，不过在构建计算图时很重要。

```python
tf.rank(rank_4_tensor)
```

```bash
<tf.Tensor: shape=(), dtype=int32, numpy=4>
```

```python
tf.shape(rank_4_tensor)
```

```txt
<tf.Tensor: shape=(4,), dtype=int32, numpy=array([3, 2, 4, 5])>
```

轴一般按从全局到局部排序：依次为 batch 维度、空间维度（width, height），feature 维度。这样可以保证 feature 向量在内存中是连续存储的。

![](images/2021-12-21-12-29-54.png)

## 4. 索引

### 4.1 单轴索引

TensorFlow 的索引遵循标准 Python 索引规则，如下：

- 以 0 开始；
- 负数索引从末尾开始倒数；
- 冒号用于切片 `start:stop:step`。

```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
```

```txt
[ 0  1  1  2  3  5  8 13 21 34]
```

用标量索引会减少维度，返回标量：

```python
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
```

```txt
First: 0
Second: 1
Last: 34
```

使用 `:` 切片则保留维度：

```python
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())
```

```txt
Everything: [ 0  1  1  2  3  5  8 13 21 34]
Before 4: [0 1 1 2]
From 4 to the end: [ 3  5  8 13 21 34]
From 2, before 7: [1 2 3 5 8]
Every other item: [ 0  1  3  8 21]
Reversed: [34 21 13  8  5  3  2  1  1  0]
```

### 4.2 多轴索引

高阶张量需要多个索引值进行索引。例如：

```python
print(rank_2_tensor.numpy())
```

```txt
[[1. 2.]
 [3. 4.]
 [5. 6.]]
```

- 对每个轴传递一个索引，返回标量

```python
# 从 2-阶 张量取出单个值
print(rank_2_tensor[1, 1].numpy())
```

```txt
4.0
```

- 组合使用整数和切片进行索引

```python
print("Second row:", rank_2_tensor[1, :].numpy()) # 第二行
print("Second column:", rank_2_tensor[:, 1].numpy()) # 第二列
print("Last row:", rank_2_tensor[-1, :].numpy()) # 最后一行
print("First item in last column:", rank_2_tensor[0, -1].numpy()) # 最后一列的第一个
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n") # 跳过第一行
```

```txt
Second row: [3. 4.]
Second column: [2. 4. 6.]
Last row: [5. 6.]
First item in last column: 2.0
Skip the first row:
[[3. 4.]
 [5. 6.]] 
```

- 对 3 阶张量进行索引

```python
>> rank_3_tensor.shape
TensorShape([3, 2, 5])
>>> print(rank_3_tensor[:, :, 4])
tf.Tensor(
[[ 4  9]
 [14 19]
 [24 29]], shape=(3, 2), dtype=int32)
```

该操作选择所有 batch，所有样本的最后一个 feature。如下图所示：

![](images/2021-12-21-14-01-23.png)

此处可以认为 batch=1, width = 3, height=2，features=5。

关于索引和切片的更多内容请参考[张量切片指南](https://tensorflow.org/guide/tensor_slicing)。

## 5. shape 操作

张量的 reshape 操作非常有用。只要总元素个数保持一致，就可以转换 shape，如下图所示：

![](images/2022-06-13-10-40-40.png)

> 将 (3x2) 张量 reshape 为其它 shape 的张量。

```python
>>> x = tf.constant([[1], [2], [3]])
>>> x.shape
TensorShape([3, 1])
```

- `.shape` 返回 `TensorShape` 对象，可以将其转换为 list

```python
>>> x.shape.as_list()
[3, 1]
```

- 可以将张量重塑为新的 shape。`tf.reshape` 操作不需要复制底层数据，因此快速且低耗

```python
>>> reshaped = tf.reshape(x, [1, 3]) # 以 list 传入的新 shape
>>> print(x.shape)
(3, 1)
>>> print(reshaped.shape)
(1, 3)
```

对 `reshape` 操作，数据在内存中保持不变，`reshape` 创建的张量具有指定 shape，和原始张量指向相同的数据。TensorFlow 使用 C-风格的内存顺序，即以行为主，将一行最右侧的索引加一，内存中只需要一步：

```python
>>> print(rank_3_tensor)
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
```

将张量展开就可以看到它在内存中的排列顺序：

```python
# 下面新的 shape [-1] 只有一维，表示转换为 rank-1 张量
# -1 表示由 TensorFlow 自动计算 axis 长度
>>> print(tf.reshape(rank_3_tensor, [-1]))
tf.Tensor(
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29], shape=(30,), dtype=int32)
```

`tf.reshape` 一般只用于合并或拆分相邻的维度。

对这个 3x2x5 张量，reshape 为 (3x2)x5 或 3x(2x5) 都是合理的，这种邻轴操作不会混淆切片：

```python
>>> print(tf.reshape(rank_3_tensor, [3 * 2, 5]), "\n")
tf.Tensor(
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]], shape=(6, 5), dtype=int32) 

>>> print(tf.reshape(rank_3_tensor, [3, -1]), "\n")
tf.Tensor(
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32) 
```

图示如下：

![](images/2021-12-21-14-34-37.png)

reshape 操作可用于任何总元素个数相同的 shape 转换，但是要遵守轴的顺序。

`tf.reshape` 不能用来交换轴，交换轴请使用 `tf.transpose`：

```python
# 错误示范

# 不能使用 reshape 重新排列轴
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")

# 输出很混乱
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# 总元素格式不对，会抛出错误
try:
    tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

```txt
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]]

 [[15 16 17 18 19]
  [20 21 22 23 24]
  [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 

tf.Tensor(
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) 

InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]
```

图示：

![](images/2021-12-21-14-33-41.png)

在 TensorFlow 中可能碰到不完全指定的形状。要么是 shape 中包含一个 `None`（对应的轴长未知），要么整个 shape 为 `None`（张量的秩未知）。

除了 [tf.RaggedTensor](#9-参差张量)，这样的 shape 只会出现在 TensorFlow 的符号化 graph 构建 API 中：

- [tf.function](https://www.tensorflow.org/guide/function)
- [keras 函数 API](https://www.tensorflow.org/guide/keras/functional)

## 6. dtype

使用 `Tensor.dtype` 属性查看 `tf.Tensor` 的数据类型。

在使用 Python 对象创建 `tf.Tensor` 时可以指定数据类型。如果不指定，TensorFlow 会根据数据自动推测类型：

- 将 Python 整数转换为 `tf.int32`；
- 将 Python 浮点数转换为 `tf.float32`；
- 其它的与 NumPy 规则一样。

类型之间可以互相转换：

```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# 转换为 uint18 会丢失小数位
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```

```txt
tf.Tensor([2 3 4], shape=(3,), dtype=uint8)
```

## 7. 广播

广播（broadcasting）是从 NumPy 借用的概念。简而言之，在特定条件下，对低维张量和高维张量进行组合操作时，低维张量会自动拉伸到高维张量的 shape，该行为称为**广播**。

最简单的广播是将一个标量和张量进行加法或乘法。此时，标量被广播为和张量相同的 shape:

```python
x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])
# 下面的三个操作结果相同
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

```txt
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
```

同样，长度为 **1** 的轴可以拉伸以匹配其它参数。在同一个运算中，进行运算的两个张量都可以拉伸。

例如，3x1 矩阵和 1x4 矩阵进行元素乘可以获得 3x4 矩阵。

```python
x = tf.reshape(x,[3,1]) # (3, 1) 广播为 (3, 4)
y = tf.range(1, 5) # (1, 4) 广播为 (3, 4)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

```txt
tf.Tensor(
[[1]
 [2]
 [3]], shape=(3, 1), dtype=int32) 

tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 

tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

![](images/2021-12-21-15-14-59.png)

下面是没有进行广播的相同操作：

```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # 运算符重载
```

```txt
tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

广播操作大多时候省时省内存，因为广播实际上没有在内存中扩展张量。

可以使用 [tf.broadcast_to](https://www.tensorflow.org/api_docs/python/tf/broadcast_to) 查看广播效果：

```python
>>> print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
tf.Tensor(
[[1 2 3]
 [1 2 3]
 [1 2 3]], shape=(3, 3), dtype=int32)
```

和前面的数学运算不同，使用 `broadcast_to` 并没有节省内存，而是真正在内存中扩展了张量。

## 8. tf.convert_to_tensor

大多数操作，如 `tf.matmul` 和 `tf.reshape` 接受 `tf.Tensor` 类型参数。不过，从前面的例子可以看出，这些操作也接受 Python 对象。

大多时候，TensorFlow 使用 `convert_to_tensor` 将非张量参数转换为张量。TF 有一个转换注册表，大多数对象，如 NumPy 的 `ndarray`，`TensorShape`，Python 列表，以及 `tf.Variable` 都会自动转换。

如果希望自定义类型能自动转换为张量，请参考 [tf.register_tensor_conversion_function](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function)。

## 9. 参差张量

某些维度的元素数量不一致的张量称为**参差张量**（ragged），使用 `tf.ragged.RaggedTensor` 创建这类张量。

例如，下面的数据无法使用常规张量表示：

![](images/2021-12-21-15-27-31.png)

```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
try:
    tensor = tf.constant(ragged_list)
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

```txt
ValueError: Can't convert non-rectangular Python sequence to Tensor.
```

但可以使用 `tf.ragged.constant` 创建 `tf.RaggedTensor`：

```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

```txt
<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
```

`tf.RaggedTensor` 的 shape 部分维度的长度未知：

```python
>>> print(ragged_tensor.shape)
(4, None)
```

## 10. 字符串张量

`tf.string` 是一个 `dtype`，换句话说，张量可以包含字符串数据。

TensorFlow 中字符串具有原子性，即将字符串看作一个整体，不能像在 Python 中那样索引，字符串的长度也不算作维度长。[tf.strings](https://www.tensorflow.org/api_docs/python/tf/strings) 中包含操作字符串张量的函数。

下面是一个字符串标量的张量：

```python
>>> scalar_string_tensor = tf.constant("Hello World")
>>> print(scalar_string_tensor)
tf.Tensor(b'Hello World', shape=(), dtype=string)
```

创建如下的 1-阶字符串张量：

![](images/2021-12-21-15-38-58.png)

```python
# 三个长度不同的字符串，也没问题
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings) # shape (3,)，不包括字符串长度
```

```txt
tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)
```

上面输出中，前缀 `b` 表示 `tf.string` dtype 不是 unicode 字符串，而是 byte-string。在 TensorFlow 中使用 Unicode 的详情请参考 [Unicode 教程](https://www.tensorflow.org/tutorials/load_data/unicode)。

也可以使用 utf-8 编码传入 unicode 字符串：

```python
tf.constant("🥳👍")
```

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>
```

`tf.strings` 包含一些基本的字符串函数。

- [tf.strings.split](https://www.tensorflow.org/api_docs/python/tf/strings/split) 拆分字符串

```python
# 可以用 split 将字符串拆分为一组张量
print(tf.strings.split(scalar_string_tensor, sep=" "))
```

```txt
tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
```

不过拆分字符串 tensor 生成的可能是 `RaggedTensor`，因为每个字符串拆分出来的长度可能不同：

```python
>>> print(tf.strings.split(tensor_of_strings))
<tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>
```

![](images/2021-12-21-15-50-02.png)

- `tf.string.to_number` 将字符串转换为数字

```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

```txt
tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)
```

- 虽然不能直接用 [tf.cast](https://www.tensorflow.org/api_docs/python/tf/cast) 将字符串 tensor 转换为数字，但可以先转换为 byte，然后再转换为数字：

```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
```

```txt
Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)
```

- 或者将字符串拆分为 unicode，然后再解码

```python
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
```

```txt
Unicode bytes: tf.Tensor(b'\xe3\x82\xa2\xe3\x83\x92\xe3\x83\xab \xf0\x9f\xa6\x86', shape=(), dtype=string)

Unicode chars: tf.Tensor([b'\xe3\x82\xa2' b'\xe3\x83\x92' b'\xe3\x83\xab' b' ' b'\xf0\x9f\xa6\x86'], shape=(5,), dtype=string)

Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)
```

dtype `tf.string` 用于所有的 raw byte 数据类型。`tf.io` 模块包含数据和 byte 之间的转换函数，包括解码 images，解析 csv 等。

## 11. 稀疏张量

TensorFlow 使用 [tf.sparse.SparseTensor](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) 表示稀疏张量，用于高效存储稀疏数据。

例如，创建如下的稀疏张量：

![](images/2021-12-21-15-58-14.png)

```python
# 稀疏张量按索引存储值，以节省内存
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# 将稀疏张量转换为密集张量
print(tf.sparse.to_dense(sparse_tensor))
```

```txt
SparseTensor(indices=tf.Tensor(
[[0 0]
 [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 

tf.Tensor(
[[1 0 0 0]
 [0 0 2 0]
 [0 0 0 0]], shape=(3, 4), dtype=int32)
```

## 12. 参考

- https://www.tensorflow.org/guide/tensor
- The TensorFlow Workshop, Mattew Moocarme & Anthony So & Anthony Maddalone
