# 变量

- [变量](#变量)
  - [1. 简介](#1-简介)
  - [2. 设置](#2-设置)
  - [3. 创建变量](#3-创建变量)
  - [4. 生命周期、命名和记录](#4-生命周期命名和记录)
  - [5. 变量和张量的存储位置](#5-变量和张量的存储位置)
  - [6. 参考](#6-参考)

Last updated: 2022-07-07, 16:59
@author Jiawei Mao
****

## 1. 简介

TensorFlow 使用变量（TF 变量）表示程序中共享、持久化的状态。下面介绍如何创建、更新和管理 [tf.Variable](../../api/tf/Variable.md)。

TF变量使用 `tf.Variable` 类创建，用于表示值可以改变的张量。高级 API 如 `tf.keras` 使用 `tf.Variable` 保存模型参数。

## 2. 设置

本笔记会讨论变量存储位置，取消注释行可以查看变量是保存在 GPU 还是 CPU。

```python
import tensorflow as tf

# 取消下面的注释可以查看变量位置，在 GPU 还是 CPU
# tf.debugging.set_log_device_placement(True)
```

## 3. 创建变量

为 `tf.Variable` 提供初始值创建 TF 变量，其 `dtype` 与初始值类型相同：

```python
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# 和 tensor 一样，variable 支持多种类型
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])
```

TF 变量外观和行为和张量一样，实际上，TF 变量底层由 `tf.Tensor` 实现。TF 变量同样具有 `dtype` 和 shape，且可以导出为 NumPy：

```python
print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())
```

```bash
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

# 下面会创建一个新的张量，TF 变量无法 reshape
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1, 4]))
```

```bash
A variable: <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[1., 2.],
       [3., 4.]], dtype=float32)>

Viewed as a tensor: tf.Tensor(
[[1. 2.]
 [3. 4.]], shape=(2, 2), dtype=float32)

Index of highest value: tf.Tensor([1 1], shape=(2,), dtype=int64)

Copying and reshaping:  tf.Tensor([[1. 2. 3. 4.]], shape=(1, 4), dtype=float32)
```

如前所述，TF 变量使用张量实现。可以使用 `tf.Variable.assign` 重新分配张量，调用 `assign` 不会创建新的张量，还是使用原张量的内存。

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

```bash
ValueError: Cannot assign value to variable ' Variable:0': Shape mismatch.The variable shape (2,), and the assigned value shape (3,) are incompatible.
```

使用和张量一样的操作对 TF 变量进行操作，一般是直接操作底层的张量。

使用已有 TF 变量创建新的TF变量，会复制底层张量。两个 TF 变量不共享内存：

```python
a = tf.Variable([2.0, 3.0])
# 使用 a 的值创建 b
b = tf.Variable(a)
a.assign([5, 6])

# a 和 b 不同
print(a.numpy())
print(b.numpy())

# 其它 assign 方法
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]
```

```sh
[5. 6.]
[2. 3.]
[7. 9.]
[0. 0.]
```

## 4. 生命周期、命名和记录

在基于 Python 的 TensorFlow 中，`tf.Variable` 实例的生命周期与其它 Python 对象一样。当没有对 TF 变量的引用时，它会自动释放。

可以对 TF 变量命名，方便记录和调试。不同 TF 变量名字可以相同：

```python
# 创建 a 和 b，名字相同，但是底层张量不同
a = tf.Variable(my_tensor, name="Mark")
# b 和 a 名字相同，但值不同，注意标量的广播
b = tf.Variable(my_tensor + 1, name="Mark")

# 两者虽然名字相同，但是元素值不同
print(a == b)
```

```sh
tf.Tensor(
[[False False]
 [False False]], shape=(2, 2), dtype=bool)
```

在保存和加载模型时，会保存 TF 变量名。模型中的 TF 变量默认会自动获取 unique 名称，因此除非有其它目的，否则不需要为 TF 变量分配名称。

虽然变量对于微分很重要，但有些变量不需要微分。可以在创建 TF 变量时设置 `trainable` 为 false 以关闭 TF 变量梯度。例如：

```python
step_counter = tf.Variable(1, trainable=False)
```

## 5. 变量和张量的存储位置

为了提高性能，TensorFlow 会尝试将张量和变量放在与其 `dtype` 兼容的最快设备上。这意味着如果有 GPU，大多数 TF 变量都会放在 GPU 上。

但是，我们可以修改该行为。下面，我们在有 GPU 的前提下将一个浮点张量和一个浮点变量放在 CPU 上。通过打开设备的日志（取消本文最上面的配置注释），可以看到 TF 变量存储位置：

```python
with tf.device('CPU:0'):

  # 创建张量
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)
```

```txt
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

可以将张量或 TF 变量放在一个设备，然后在另一个设备上计算。这样会带来延迟，因为需要在不同设备之间复制数据。

但是，如果有多个 GPU，但只需要一个变量副本，就可以这么干：

```python
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)
```

```txt
tf.Tensor(
[[ 1.  4.  9.]
 [ 4. 10. 18.]], shape=(2, 3), dtype=float32)
```

有关分布式训练的更多信息，请参考 [分布式训练指南](../distributed_training.md)。

## 6. 参考

- https://www.tensorflow.org/guide/variable
