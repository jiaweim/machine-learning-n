# tf.range

2022-02-23, 15:28
****

## 简介

```python
tf.range(limit, delta=1, dtype=None, name='range')
tf.range(start, limit, delta=1, dtype=None, name='range')
```

以增量 `delta` 在 `[start,limit)` 之间生成数字序列。

除非通过 `dtype` 显式设置，否则生成的张量类型通过输入推导。

和Python 内置的 `range` 一样，`start` 默认为 0，所以 `range(n) = range(0, n)`。

> 和 np.arange 功能一致。

## 参数

|参数|说明|
|---|---|
|start|A 0-D Tensor (scalar). Acts as first entry in the range if limit is not None; otherwise, acts as range limit and first entry defaults to 0.|
|limit|A 0-D Tensor (scalar). Upper limit of sequence, exclusive. If None, defaults to the value of start while the first entry of the range defaults to 0.|
|delta|A 0-D Tensor (scalar). Number that increments start. Defaults to 1.|
|dtype|The type of the elements of the resulting tensor.|
|name|A name for the operation. Defaults to "range".|

## 示例

```python
start = 3
limit = 18
delta = 3
print(tf.range(start, limit, delta))
```

```sh
tf.Tensor([ 3  6  9 12 15], shape=(5,), dtype=int32)
```

因为 `start`、`limit` 和 `delta` 都是整数，所以生成的序列也是整数。

```python
start = 3
limit = 1
delta = -0.5

print(tf.range(start, limit, delta))
```

```sh
tf.Tensor([3.  2.5 2.  1.5], shape=(4,), dtype=float32)
```

由于 `delta` 是浮点数，所以生成的序列是浮点类型。因为 `start` 大于 `limit`，`delta` 是负数，所以生成的是降序序列。

```python
>>> limit = 5
>>> tf.range(limit)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4])>
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/range
