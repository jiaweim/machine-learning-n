# tf.reshape

Last updated: 2022-09-26, 15:36
****

## 简介

```python
tf.reshape(
    tensor, shape, name=None
)
```

重塑（reshape）张量。

给定 `tensor`，该操作返回一个新的 `tf.Tensor`，与原张量具有相同的值（顺序也相同），但是形状（`shape`）不同。

```python
>>> t1 = [[1, 2, 3],
...       [4, 5, 6]]
>>> print(tf.shape(t1).numpy())
[2 3]
>>> t2 = tf.reshape(t1, [6])
>>> t2
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6])>
>>> tf.reshape(t2, [3, 2])
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4],
       [5, 6]])>
```

`tf.reshape` 不改变张量中元素的个数和顺序，因此可以重用底层缓存数据。因此 `tf.reshape` 操作很快，与操作的张量大小无关。

```python
>>> tf.reshape([1, 2, 3], [2, 2])
InvalidArgumentError                      Traceback (most recent call last)
...
InvalidArgumentError: Input to reshape is a tensor with 3 values, but the requested shape has 4
```

- 如果要对数据重新排序来实现维度转换，则需要使用 `tf.transpose`。

```python
>>> t = [[1, 2, 3],
...      [4, 5, 6]]
>>> tf.reshape(t, [3, 2]).numpy()
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> tf.transpose(t, perm=[1, 0]).numpy()
array([[1, 4],
       [2, 5],
       [3, 6]])
```

- 如果参数 `shape` 某个值为 `-1`，表示该维度根据总元素个数保持不变的原则进行计算。特别是，`shape` 为 `[-1]` 表示转换为 1-D。`shape` 最多只能有一个值为 -1。

```python
>>> t = [[1, 2, 3],
...      [4, 5, 6]]
>>> tf.reshape(t, [-1]) # 转换为 1-D
<tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6])>
>>> tf.reshape(t, [3, -1])
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4],
       [5, 6]])>
>>> tf.reshape(t, [-1, 2])
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4],
       [5, 6]])>       
```

- `tf.reshape(t, [])` 将包含一个元素的张量 `t` 转换为标量

```python
>>> tf.reshape([7], []).numpy()
7
```

- (9,) 到 (3,3)

```python
>>> t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> print(tf.shape(t).numpy())
[9]
>>> tf.reshape(t, [3, 3])
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])>
```

- (2,2,2) 到 (2,4)

```python
>>> t = [[[1, 1], [2, 2]],
...      [[3, 3], [4, 4]]]
>>> print(tf.shape(t).numpy())
[2 2 2]
>>> tf.reshape(t, [2, 4])
<tf.Tensor: shape=(2, 4), dtype=int32, numpy=
array([[1, 1, 2, 2],
       [3, 3, 4, 4]])>
```

- -1 的各种应用

```python
>>> t = [[[1, 1, 1],
...       [2, 2, 2]],
...      [[3, 3, 3],
...       [4, 4, 4]],
...      [[5, 5, 5],
...       [6, 6, 6]]]
>>> print(tf.shape(t).numpy())
[3 2 3]
>>> # -1 表示转换为 1-D
>>> tf.reshape(t, [-1])
<tf.Tensor: shape=(18,), dtype=int32, numpy=array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])>
>>> # -1 用来自动计算该维度大小，这里为 9
>>> tf.reshape(t, [2, -1])
<tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
       [4, 4, 4, 5, 5, 5, 6, 6, 6]])>
>>> # -1 用来自动计算该维度大小，这里为 2
>>> tf.reshape(t, [-1, 9])
<tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
       [4, 4, 4, 5, 5, 5, 6, 6, 6]])>
>>> # -1 这里自动计算为 3
>>> tf.reshape(t, [2, -1, 3])
<tf.Tensor: shape=(2, 3, 3), dtype=int32, numpy=
array([[[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]],

       [[4, 4, 4],
        [5, 5, 5],
        [6, 6, 6]]])>
```

## 参数

|参数|说明|
|---|---|
|tensor|A Tensor|
|shape|A Tensor. 类型为 `int32` 或 `int64` 定义输出 tensor 的 shape|
|name|操作名称|

返回 `Tensor`，与 `tensor` 类型相同。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/reshape
