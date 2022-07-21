# 梯度和自动微分

- [梯度和自动微分](#梯度和自动微分)
  - [简介](#简介)
  - [计算梯度](#计算梯度)
    - [Gradient tapes](#gradient-tapes)
    - [模型梯度](#模型梯度)
    - [设置 tape 的记录内容](#设置-tape-的记录内容)
  - [中间变量梯度](#中间变量梯度)
  - [性能](#性能)
  - [非标量的梯度](#非标量的梯度)
  - [控制流](#控制流)
  - [计算 None 的梯度](#计算-none-的梯度)
    - [1. TF 变量替换为 tensor](#1-tf-变量替换为-tensor)
    - [2. 在 TensorFlow 外执行计算](#2-在-tensorflow-外执行计算)
    - [3. 通过整数或字符串获取梯度](#3-通过整数或字符串获取梯度)
    - [4. 使用状态对象计算梯度](#4-使用状态对象计算梯度)
  - [梯度注册](#梯度注册)
  - [用 0 替代 None](#用-0-替代-none)
  - [参考](#参考)

2021-12-21, 19:16
@author Jiawei Mao
****

## 简介

[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation) 对神经网络训练所用的反向传播算法十分重要。

下面介绍 TensorFlow 计算梯度的方法，特别是即时执行（eager execution）。

代码配置：

```python
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
```

## 计算梯度

为了自动微分，TensorFlow 需要记住正向传播的操作步骤。然后，在反向传播时以相反的顺序遍历这些操作来计算梯度。

### Gradient tapes

TensorFlow 使用 [tf.GradientTape](../../api/tf/GradientTape.md) API 执行自动微分，即根据输入（一般为 `tf.Variable`）计算梯度。TensorFlow 会记录在 `tf.GradientTape` 上下文中执行的操作，然后在反向传播时使用 tape 记录的信息计算梯度。

下面是一个简单示例：

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape: # 下面的操作都会被记住
    y = x ** 2
```

记住这些操作后，使用 `GradientTape.gradient(target, sources)` 计算指定目标（target，一般是损失值）相对某个变量的梯度。

```python
>>> # dy = 2x * dx
>>> dy_dx = tape.gradient(y, x)
>>> dy_dx.numpy()
6.0
```

这里是对标量计算梯度，对张量计算梯度的操作一样：

```python
w = tf.Variable(tf.random.normal((3, 2)), name='w')  # shape (3,2)
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]  # shape (1,3)

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)
```

要计算损失值对两个变量的梯度，将它们都传入 `gradient` 方法的 `sources` 参数。tape 对传入的 `sources` 参数类型非常灵活，接受 dict 或 list 的嵌套组合，并按相同结构返回梯度。

```python
[dl_dw, dl_db] = tape.gradient(loss, [w, b])
```

每个 source 的梯度与 source 的 shape 相同：

```python
>>> print(w.shape)
(3, 2)
>>> print(dl_dw.shape)
(3, 2)
```

下面传入 dict 类型计算梯度：

```python
my_vars = {
    'w': w,
    'b': b
}
grad = tape.gradient(loss, my_vars)
grad['b']
```

```sh
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([-2.9555566, -0.3094782], dtype=float32)>
```

### 模型梯度

`tf.Module` 及其子类（`layers.Layer`, `keras.Model`）一般包含许多 `tf.Variables`，用于[检查点](../checkpoint.md)和[导出模型](../saved_model.md)。

大多时候需要计算模型可训练变量的梯度。由于 `tf.Module` 及其子类将这些变量都放在 `Module.trainable_variables` 属性中，因此可以很容易地计算这些梯度：

```python
layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
  # 向前传播
  y = layer(x)
  loss = tf.reduce_mean(y**2)

# 对所有可训练变量计算梯度
grad = tape.gradient(loss, layer.trainable_variables)
```

```python
for var, g in zip(layer.trainable_variables, grad):
  print(f'{var.name}, shape: {g.shape}')
```

```sh
dense/kernel:0, shape: (3, 2)
dense/bias:0, shape: (2,)
```

### 设置 tape 的记录内容

tape 默认记录可训练 `tf.Variable` 的所有操作。原因如下：

- tape 需要知道在向前传播时记录哪些操作，以便在向后传播时计算梯度；
- tape 还需要保存中间步骤的输出结果，因此应该避免记录不必要的操作，以减少开销；
- 一般计算模型的损失值相对所有可训练变量的梯度。

例如，下面计算梯度会失败，因为 tape 默认不记录 `tf.Tensor`，且部分 `tf.Variable` 不可训练：

```python
# 可训练变量
x0 = tf.Variable(3.0, name='x0')
# 不可训练
x1 = tf.Variable(3.0, name='x1', trainable=False)
# 非变量：TF 变量+tensor 返回 tensor 类型
x2 = tf.Variable(2.0, name='x2') + 1.0
# 非变量
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
    y = (x0 ** 2) + (x1 ** 2) + (x2 ** 2)

grad = tape.gradient(y, [x0, x1, x2, x3])

for g in grad:
    print(g)
```

```sh
tf.Tensor(6.0, shape=(), dtype=float32)
None
None
None
```

可以使用 `GradientTape.watched_variables` 方法查看 tape 监视的变量。

```python
>>> [var.name for var in tape.watched_variables()]
['x0:0']
```

`tf.GradientTape` 提供了钩子函数用于控制监控哪些变量。

例如，如果你想要记录 `tf.Tensor` 的梯度，可以调用 `GradientTape.watch(x)` 方法：

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
  tape.watch(x)
  y = x**2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())
```

```sh
6.0
```

相反，如果不需要 tape 记录所有 `tf.Variable`，可以在创建 tape 时设置 `watch_accessed_variables=False` 修改该默认行为。例如，下面创建两个 TF 变量，但是只计算一个变量的梯度：

```python
x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(x1) # 记录 x1
  y0 = tf.math.sin(x0)
  y1 = tf.nn.softplus(x1)
  y = y0 + y1
  ys = tf.reduce_sum(y)
```

由于没有对 `x0` 调用 `GradientTape.watch`，所以无法计算其梯度：

```python
# dys/dx1 = exp(x1) / (1 + exp(x1)) = sigmoid(x1)
grad = tape.gradient(ys, {'x0': x0, 'x1': x1})

print('dy/dx0:', grad['x0'])
print('dy/dx1:', grad['x1'].numpy())
```

```sh
dy/dx0: None
dy/dx1: 0.9999546
```

## 中间变量梯度

可以计算输出结果对 `tf.GradientTape` 上下文中定义的中间变量的梯度：

```python
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
    z = y * y
# 计算 z 相对中间变量 y 的梯度
# dz_dy = 2y, y = x ** 2 = 9
print(tape.gradient(z, y).numpy())
```

```sh
18.0
```

`GradientTape` 持有的资源默认在调用 `GradientTape.gradient` 后释放。如果想多次调用 `GradientTape.gradient` 以多次计算梯度，可以设置 `persistent=True`，这样 tape 对象持有的资源在 tape 对象被垃圾回收时才释放。例如：

```python
x = tf.constant([1, 3.0])
with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  y = x * x
  z = y * y

print(tape.gradient(z, x).numpy())  # [4.0, 108.0] (4 * x**3 at x = [1.0, 3.0])
print(tape.gradient(y, x).numpy())  # [2.0, 6.0] (2 * x at x = [1.0, 3.0])
```

```sh
[  4. 108.]
[2. 6.]
```

然后，在不需要时删掉对 tape 的引用：

```python
del tape   # 删除对 tape 的引用
```

## 性能

在 tape 上下文中进行操作会有一点额外开销，虽然对大多数 eager 执行来说开销不大，但是还是应该只在需要时使用。

`GradientTape` 存储中间结果需要消耗内存，包括每层网络的输入和输出，用于反向传播。

为了提高效率，部分操作（如 `ReLU`）的中间结果不保留。但是，如果设置 `persistent=True`，则不会舍弃任何内容，内存使用峰值会高一些。

## 非标量的梯度

梯度从根本上来说，是对标量的操作：

```python
x = tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
  y0 = x**2
  y1 = 1 / x

print(tape.gradient(y0, x).numpy())
print(tape.gradient(y1, x).numpy())
```

```sh
4.0
-0.25
```

因此，如果要计算多个 target 的梯度，对每个 source 来说，结果为：

- target 加和的梯度；
- 每个 target 梯度的加和。

两个结果是一样的。例如，计算梯度和：

```python
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
  y0 = x**2
  y1 = 1 / x

print(tape.gradient({'y0': y0, 'y1': y1}, x).numpy())
```

```sh
3.75
```

类似地，如果 target 不是标量，则计算和的梯度：

```python
x = tf.Variable(2.)

with tf.GradientTape() as tape:
  y = x * [3., 4.]

print(tape.gradient(y, x).numpy())
```

```sh
7.0
```

因此，计算多个损失的加和的梯度非常简单。

如果需要每项的单独梯度，请参考 [Jacobians](../advanced_autodiff.md#jacobians)。

在某些情况下，可以跳过 Jacobian。对逐元素计算，sum 的梯度给出了每个元素对其输入元素的导数，因为每个元素都是独立：

```python
x = tf.linspace(-10.0, 10.0, 200+1)

with tf.GradientTape() as tape:
  tape.watch(x)
  y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y, x)

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
_ = plt.xlabel('x')
```

![](images/2021-12-23-19-10-55.png)

## 控制流

因为 gradientTape 在执行时记录操作，所以很自然地能够处理控制流（如 `if` 和 `while` 语句）。

下面对 `if` 的每个分支使用不同的变量，梯度只和实际使用的变量关联：

```python
x = tf.constant(1.0)

v0 = tf.Variable(2.0)
v1 = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  if x > 0.0:
    result = v0
  else:
    result = v1**2 

dv0, dv1 = tape.gradient(result, [v0, v1])

print(dv0)
print(dv1)
```

```sh
tf.Tensor(1.0, shape=(), dtype=float32)
None
```

因为控制语句本身是不可微的，所以它们对基于梯度的优化器不可见。

根据 `x` 的值不同，tap 记录 `result = v0` 或 `result = v1**2`。对 `x` 梯度总是 `None`：

```python
>>> dx = tape.gradient(result, x)
>>> print(dx)
None
```

## 计算 None 的梯度

当 target 没有连接到 source 时，梯度为 `None`。例如：

```python
x = tf.Variable(2.)
y = tf.Variable(3.)

with tf.GradientTape() as tape:
  z = y * y
print(tape.gradient(z, x))
```

```sh
None
```

这里 `z` 和 `x` 很明显无关，但有些不是很明显的情况会导致梯度断开。

### 1. TF 变量替换为 tensor

tape 会自动记录 `tf.Variable`，但默认不记录 `tf.Tensor`。

一个常见的错误就是没有用 `Variable.assign` 更新变量，无意中导致 `tf.Variable` 替换为 `tf.Tensor`。例如：

```python
x = tf.Variable(2.0)

for epoch in range(2):
  with tf.GradientTape() as tape:
    y = x+1

  print(type(x).__name__, ":", tape.gradient(y, x))
  x = x + 1   # 这里应该用 `x.assign_add(1)`
```

```sh
ResourceVariable : tf.Tensor(1.0, shape=(), dtype=float32)
EagerTensor : None
```

### 2. 在 TensorFlow 外执行计算

在 TensorFlow 之外执行的计算 tape 无法计算梯度。例如：

```python
x = tf.Variable([[1.0, 2.0],
                 [3.0, 4.0]], dtype=tf.float32)

with tf.GradientTape() as tape:
  x2 = x**2

  # 这一步用 NumPy 计算的
  y = np.mean(x2, axis=0)

  # `reduce_mean` 操作会使用 `tf.convert_to_tensor`
  # 将 NumPy 数组转换为 constant tensor
  y = tf.reduce_mean(y, axis=0)

print(tape.gradient(y, x))
```

```sh
None
```

### 3. 通过整数或字符串获取梯度

整数和字符串不可微。如果计算路径中使用了这类数据类型，则没有梯度。

字符串很容易避免。但是如果不指定 `dtype`，很容易创建 `int` 变量：

```python
x = tf.constant(10)

with tf.GradientTape() as g:
  g.watch(x)
  y = x * x

print(g.gradient(y, x))
```

```sh
WARNING:tensorflow:The dtype of the watched tensor must be floating (e.g. tf.float32), got tf.int32
WARNING:tensorflow:The dtype of the target tensor must be floating (e.g. tf.float32) when calling GradientTape.gradient, got tf.int32
WARNING:tensorflow:The dtype of the source tensor must be floating (e.g. tf.float32) when calling GradientTape.gradient, got tf.int32
None
```

TensorFlow 不会自动强制类型转换，因此，在实际使用中，往往会得到一个类型错误，而不是缺失梯度错误。

### 4. 使用状态对象计算梯度

状态（state）会终止梯度。对内含状态的对象，tape 只能看到当前状态，而不知道获得该状态的过程。

`tf.Tensor` 是 immutable，创建后无法修改。它包含值，但是没有状态（state）。上面讨论的所有操作都没有状态，例如 `tf.matmul` 的输出仅取决于其输入。

`tf.Variable` 有内部状态，即其值（value）。当使用 TF 变量时，读取其状态。计算一个变量的梯度是常规操作，但是变量的状态会阻碍梯度的计算。例如：

```python
x0 = tf.Variable(3.0)
x1 = tf.Variable(0.0)

with tf.GradientTape() as tape:
  # 更新 x1 = x1 + x0.
  x1.assign_add(x0)
  # tape 从 x1 开始记录
  y = x1**2   # y = (x1 + x0)**2

# 行不通
print(tape.gradient(y, x0))   #dy/dx0 = 2*(x1 + x0)
```

```sh
None
```

类似地，`tf.data.Dataset` 迭代器和 `tf.queue` 也是有状态的，它们会终止所有通过它们的变量的梯度。

## 梯度注册

有些 `tf.Operation` 操作被注册为不可微分，梯度返回 `None`，有些则没有注册。

[tf.raw_ops](https://www.tensorflow.org/api_docs/python/tf/raw_ops)显示有哪些低级操作注册了梯度。

如果试图通过一个没有注册梯度的浮点操作计算梯度，tape 会直接抛出错误，而不是返回 `None`。

例如，`tf.image.adjust_contrast` 函数封装了 `raw_ops.AdjustContrastv2`，它可以有梯度，但是还没有实现，即没有注册：

```python
image = tf.Variable([[[0.5, 0.0, 0.0]]])
delta = tf.Variable(0.1)

with tf.GradientTape() as tape:
  new_image = tf.image.adjust_contrast(image, delta)

try:
  print(tape.gradient(new_image, [image, delta]))
  assert False   # This should not happen.
except LookupError as e:
  print(f'{type(e).__name__}: {e}')
```

```sh
LookupError: gradient registry has no entry for: AdjustContrastv2
```

如果需要对这个操作进行微分，要么实现这个梯度并使用 `tf.RegisterGradient` 注册，或者用其它注册过的函数重新实现。

## 用 0 替代 None

对无法计算的梯度，用 0 替代返回值 `None` 有时候很方便。使用 `gradient()` 的 `unconnected_gradients` 参数可以设置无法计算梯度时的返回值：

```python
x = tf.Variable([2., 2.])
y = tf.Variable(3.)

with tf.GradientTape() as tape:
  z = y**2
print(tape.gradient(z, x, unconnected_gradients=tf.UnconnectedGradients.ZERO))
```

```sh
tf.Tensor([0. 0.], shape=(2,), dtype=float32)
```

## 参考

- https://www.tensorflow.org/guide/autodiff
