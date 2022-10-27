# tf.constant

Last updated: 2022-10-27, 19:52

****

## 简介

```python
tf.constant(
    value, dtype=None, shape=None, name='Const'
)
```

使用 tensor-like 对象创建常量 tensor。

> **Note:** 所有 eager `tf.Tensor` 的值不可变（与 `tf.Variable` 相反）。从 `tf.constant` 返回的 tensor 没有特别的地方，与 tf.convert_to_tensor 本质上没有区别。之所以称为 `tf.constant` 是因为 `value` 嵌入到 `tf.Graph` 的 Const node。`tf.constant` 用于断言 value 能以这种方式嵌入。

- 如果没有指定参数 `dtype`，则从 `value` 的类型推断：

```python
# 从 list 创建 1-D 常量 tensor
>>> tf.constant([1, 2, 3, 4, 5, 6])
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6])>
>>> 从 numpy 数组创建 2-D 常量 tensor
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> tf.constant(a)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
```

- 如果指定了 `dtype`，则生成的 tensor 值被强制转换为 `dtype`

```python
>>> tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)
<tf.Tensor: shape=(6,), dtype=float64, numpy=array([1., 2., 3., 4., 5., 6.])>
```

- 如果设置了 `shape`，则将 `value` reshape 为对应 `shape`，标量值通过广播进行填充

```python
>>> tf.constant(0, shape=(2, 3))
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[0, 0, 0],
       [0, 0, 0]])>
>>> tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
```

> 从该示例可以看出，`shape` 支持 tuple 和 list 类型

- 将 eager tensor 传入 `value` 没有效果，甚至还可以传递梯度

```python
>>> v = tf.Variable([0.0])
>>> with tf.GradientTape() as g:
...     loss = tf.constant(v + v)
>>> g.gradient(loss, v).numpy()
array([2.], dtype=float32)
```

但是，由于 `tf.constant` 将 `value` 嵌入到 `tf.Graph`，因此对符号 tensor 会失败：

```python
>>> with tf.compat.v1.Graph().as_default():
...     i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
...     t = tf.constant(i)
Traceback (most recent call last)
...
TypeError: ...
```

`tf.constant` 将在当前设备上创建张量。如果输入已是张量，则保持其位置不变。

**相关操作：**

- `tf.convert_to_tensor` 与 `tf.constant` 类似，但是
  - `tf.convert_to_tensor` 没有 `shape` 参数
  - 支持符号张量

```python
>>> with tf.compat.v1.Graph().as_default():
...     i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)
...     t = tf.convert_to_tensor(i)
```

- `tf.fill`，主要差别：
  - `tf.constant` 支持任意张，而 `tf.fill` 只支持 uniform 标量张量
  - `tf.fill` 创建的 TF 操作在 graph 中运行时展开，因此可以有效表示大型张量
  - 由于 t`f.fill` 不嵌入值，所以可以产生动态大小输出。

## 参数

|参数|说明|
|---|---|
|`value`|`dtype` 类型的常量值或 list|
|`dtype`|生成张量的元素类型|
|`shape`|（可选）生成张量的维度|
|`name`|(可选) tensor 名称|

返回常量张量。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/constant
