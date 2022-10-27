# tf.constant

***

## 简介

```python
tf.constant(
    value, dtype=None, shape=None, name='Const'
)
```

使用 tensor-like 对象创建常量 tensor。

> **Note:** 所有 eager `tf.Tensor` 的值不可变（与 `tf.Variable` 相反）。从 tf.constant 返回的 tensor 没有特别 constant 的地方，与 tf.convert_to_tensor 本质上没有区别。

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

- 将 eager tensor 传入 `value` 没有效果，甚至可以传递梯度

```python
>>> v = tf.Variable([0.0])
>>> with tf.GradientTape() as g:
>>>     loss = tf.constant(v + v)
>>> g.gradient(loss, v).numpy()
array([2.], dtype=float32)
```

## 参数

|参数|说明|
|---|---|
|`value`|A constant value (or list) of output type dtype.|
|`dtype`|The type of the elements of the resulting tensor.|
|`shape`|Optional dimensions of resulting tensor.|
|`name`|(可选) tensor 名称|

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/constant
