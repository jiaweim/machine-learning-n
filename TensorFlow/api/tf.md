# tf

- [tf](#tf)
  - [tf.constant](#tfconstant)

2021-11-10, 09:17
***

## tf.constant

```py
tf.constant(
    value, dtype=None, shape=None, name='Const'
)
```

使用类张量对象创建常量 tensor。

如果没有指定 `dtype` 参数，则从 `value` 推断类型。

**例1**，从 list 创建 1-D Tensor

```py
>>> import tensorflow as tf
>>> tf.constant([1, 2, 3, 4, 5, 6])
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6])>
```

**例2**，从 numpy array 创建 tensor

```py
>>> import numpy as np
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> tf.constant(a)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
```

如果指定 `dtype`，则生成的 tensor 值转换为对应类型。

**例3**，指定 `dtype`

```py
>>> tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)
<tf.Tensor: shape=(6,), dtype=float64, numpy=array([1., 2., 3., 4., 5., 6.])>
```

如果指定 `shape`，则 `value` 会转换为对应 shape。标量值则会扩展到对应 shape。

**例4**，指定 `shape`

```py
>>> tf.constant(0, shape=(2, 3))
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[0, 0, 0],
       [0, 0, 0]])>
>>> tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
```

如果传入 `value` 的类型为即时（eager） Tensor，则 `tf.constant` 无效，甚至还会传递梯度：

**例5**，传递即时张量

```py
>>> v = tf.Variable([0.0])
>>> with tf.GradientTape() as g:
        loss = tf.constant(v + v)
>>> g.gradient(loss, v).numpy()
array([2.], dtype=float32)
```

