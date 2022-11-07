# tf.one_hot

Last updated: 2022-10-28, 13:11
****

## 简介

```python
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```

> **aliases:** `tf.compat.v1.one_hot`

返回一个 one-hot 张量。

`indices` 中索引表示的位置填充 `on_value`，其它位置填充 `off_value`。

- `on_value` 和 `off_value` 的数据类型必须匹配。如果指定了 `dtype`，则它们必须与 `dtype` 类型相同。
- 如果没有指定 `on_value`，则取 `dtype` 类型的默认值 1
- 如果没有指定 `off_value`，则取 `dtype` 类型的默认值 0
- 如果输入 `indices` 的秩为 `N`，则输出秩将为 `N+1`。新轴在 `axis` 维度创建，即新轴默认为最后一个维度。
- 如果 `indices` 为标量，则输出为长度为 `depth` 的向量
- 如果 `indices` 是长度为 `features` 的向量，则输出 shape 为

```python
(features, depth) if axis == -1
(depth, features) if axis == 0
```

- 如果 `indices` 是 shape 为 `[batch, features]` 的矩阵，则输出 shape 为

```python
(batch, features, depth) if axis == -1
(batch, depth, features) if axis == 1
(depth, batch, features) if axis == 0
```

- 如果 `indices`是 `RaggedTensor`，`axis` 参数必须为正数，并指向一个非 ragged 轴。输出相当于在 `RaggedTensor` 的值上应用 `one_hot`，并从输出结果创建一个新的 `RaggedTensor`。
- 如果没指定 `dtype`，则从 `on_value` 或 `off_value` 的数据类型推断类型。如果 `on_value`, `off_value` 以及 `dtype` 都没指定，则默认为 `tf.float32`。

> **Note:** 如果需要非数字类型输出，如 `tf.string`, `tf.bool` 等，则必须同时指定 `on_value` 和 `off_value`。

例如：

```python
>>> indices = [0, 1, 2]
>>> depth = 3
>>> tf.one_hot(indices, depth)  # 输出: [3 x 3]
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]], dtype=float32)>
```

```python
>>> indices = [0, 2, -1, 1]
>>> depth = 3
>>> tf.one_hot(indices, depth,
...            on_value=5.0,
...            off_value=0.0,
...            axis=-1)  # output: [4 x 3]
<tf.Tensor: shape=(4, 3), dtype=float32, numpy=
array([[5., 0., 0.], # one_hot(0)
       [0., 0., 5.], # one_hot(2)
       [0., 0., 0.], # one_hot(-1)
       [0., 5., 0.]], dtype=float32)> # # one_hot(1)
```

> 负数索引用 `off_value` 填充

```python
>>> indices = [[0, 2], [1, -1]]
>>> depth = 3
>>> tf.one_hot(indices, depth,
...            on_value=1.0, off_value=0.0,
...            axis=-1)  # output: [2 x 2 x 3]
<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[1., 0., 0.],
        [0., 0., 1.]],

       [[0., 1., 0.],
        [0., 0., 0.]]], dtype=float32)>
```

```python
>>> indices = tf.ragged.constant([[0, 1], [2]])
>>> depth = 3
>>> tf.one_hot(indices, depth)  # output: [2 x None x 3]
<tf.RaggedTensor [[[1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]>
```

## 参数

|参数|说明|
|---|---|
|indices|索引张量|
|depth|标量：定义 one-hot 维度深度|
|on_value|标量：当 `indices[j] = i` 时的填充值，默认 1|
|off_value|标量：当 `indices[j] != i` 时的填充值，默认 0|
|axis|要填充的维度，默认 -1，表示最内侧新的维度|
|dtype|输出张量类型|
|name|（可选）操作名称|

**Returns**

one-hot 张量。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/one_hot
