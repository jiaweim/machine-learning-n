# tf.repeat

2022-02-23, 16:24
****

## 简介

```python
tf.repeat(
    input, repeats, axis=None, name=None
)
```

将指定 `axis` 的元素重复 `repeats` 次，返回和 `input` 除了指定 `axis`其它 shape 相同的张量。

如果 `axis` 为 None，则将数组拉平为 1 维，然后各个元素分别重复指定次数。

## 参数

|参数|说明|
|---|---|
|input|An N-dimensional Tensor.|
|repeats|An 1-D int Tensor. The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis. len(repeats) must equal input.shape[axis] if axis is not None.|
|axis|An int. The axis along which to repeat values. By default (axis=None), use the flattened input array, and return a flat output array.|
|name|A name for the operation.|

## 示例

```python
>>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
<tf.Tensor: shape=(5,), dtype=string, numpy=array([b'a', b'a', b'a', b'c', b'c'], dtype=object)>
>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0) # 2x2 变为 5x2
<tf.Tensor: shape=(5, 2), dtype=int32, numpy=
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4],
       [3, 4]])>
>>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1) # 2x2 变为 2x5
<tf.Tensor: shape=(2, 5), dtype=int32, numpy=
array([[1, 1, 2, 2, 2],
       [3, 3, 4, 4, 4]])>
>>> repeat(3, repeats=4) # 重复单个值
<tf.Tensor: shape=(4,), dtype=int32, numpy=array([3, 3, 3, 3])>
>>> repeat([[1,2], [3,4]], repeats=2)
<tf.Tensor: shape=(8,), dtype=int32, numpy=array([1, 1, 2, 2, 3, 3, 4, 4])>
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/repeat
