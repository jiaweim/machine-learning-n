# 变量

- [变量](#变量)
  - [简介](#简介)
  - [创建变量](#创建变量)
  - [生命周期、命名和监视](#生命周期命名和监视)
  - [变量和张量位置](#变量和张量位置)
  - [参考](#参考)

2021-12-21, 18:28
***

## 简介

推荐使用 TensorFlow 变量（TF 变量）表示程序中共享、持久状态。下面介绍如何创建、更新和管理 `tf.Variable`。

TF变量通过 `tf.Variable` 类创建。`tf.Variable` 表示值可以改变的张量。高级 API 如 `tf.keras` 使用 `tf.Variable` 保存模型参数。

初始配置：

```python
import tensorflow as tf

# 取消下面的注释可以查看变量位置
# tf.debugging.set_log_device_placement(True)
```

## 创建变量

为 `tf.Variable` 提供初始值创建TF变量，其 `dtype` 与初始值相同：

```python
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# 和 tensor 一样，variable 可以是各种类型
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])
```

TF 变量看起来和张量一样，实际上，TF 变量是由 `tf.Tensor` 支持的数据结构。和张量一样，TF 变量具有 `dtype` 和 shape，并可以导出为 NumPy：

```python
print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())
```

```sh
Shape:  (2, 2)
DType:  <dtype: 'float32'>
As NumPy:  [[1. 2.]
 [3. 4.]]
```

TF 变量大多数操作和张量一样，不过不支持 reshape：

```python
print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# 下面创建一个新的张量，TF 变量无法 reshape
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1, 4]))
```

```sh
A variable: <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>

Viewed as a tensor: tf.Tensor(
[[1. 2.]
 [3. 4.]], shape=(2, 2), dtype=float32)

Index of highest value: tf.Tensor([1 1], shape=(2,), dtype=int64)

Copying and reshaping:  tf.Tensor([[1. 2. 3. 4.]], shape=(1, 4), dtype=float32)
```

如上所示，TF 变量由张量保存数组。可以使用 `tf.Variable.assign` 重新分配张量，调用 `assign` 不会创建新的张量，而是使用原有张量所在内存。

```python
a = tf.Variable([2.0, 3.0])
# 使用相同的 dtype, float32
a.assign([1, 2]) 
# 下面的操作不允许，因为会 resize TF 变量 
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

```sh
ValueError: Cannot assign value to variable ' Variable:0': Shape mismatch.The variable shape (2,), and the assigned value shape (3,) are incompatible.
```

使用和张量一样的操作对TF变量进行操作，一般是直接操作底层的张量。

从已有TF变量创建新的TF变量，会复制底层张量。两个TF变量不会共享相同内存：

```python
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]
```

```sh
[5. 6.]
[2. 3.]
[7. 9.]
[0. 0.]
```

## 生命周期、命名和监视

在基于 Python 的 TensorFlow 中，`tf.Variable` 实例具有与其它 Python 对象一样的生命周期。当没有对 TF 变量的引用时，它会自动 deallocated。

可以对 TF 变量命名，这样方便跟踪和调试。可以给两个 TF 变量相同名字：

```python
# 创建 a 和 b，名字相同，但是底层张量不同
a = tf.Variable(my_tensor, name="Mark")
# b 和 a 名字相同，但是值不同，注意标量的广播
b = tf.Variable(my_tensor + 1, name="Mark")

# 两者虽然名字相同，但是元素值不同
print(a == b)
```

```sh
tf.Tensor(
[[False False]
 [False False]], shape=(2, 2), dtype=bool)
```

在保存和加载模型时，TF 变量名称被保留。默认情况下，TF 变量会自动获取 unique 变量名称，因此除非有其它目的，否则不需要自动为 TF 变量分配名称。

虽然变量对于微分很重要，但有些变量不需要微分。可以通过在创建 TF 变量时设置 `trainable` 为 false 关闭TF 变量梯度。例如：

```python
step_counter = tf.Variable(1, trainable=False)
```

## 变量和张量位置

为了获得更好的性能，TensorFlow 会尝试将张量和变量放在与其 `dtype` 兼容的最快的设备上。这意味着如果有 GPU，大多数 TF 变量都会放置在 GPU 上。

但是，我们可以修改该行为。下面，我们在有 GPU 的前提下将一个浮点张量和一个浮点变量放在 CPU 上。通过打开设备的日志纪录（本文最上面的配置），可以看到TF 变量放在哪：

```python
with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)
```

```sh
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

可以将张量或TF变量放在一个设备，计算在另一个设备。这样会带来延迟，因为数据需要在不同设备之间复制。

但是，如果你有多个 GPU，但只需要一个变量副本，就可以这么干：

```python
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)
```

```sh
tf.Tensor(
[[ 1.  4.  9.]
 [ 4. 10. 18.]], shape=(2, 3), dtype=float32)
```

## 参考

- https://www.tensorflow.org/guide/variable
