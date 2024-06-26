# reduce_sum

Last updated: 2022-09-26, 17:24
****

## 简介

```python
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)
```

沿指定维度计算张量元素的加和。

这是 `tf.math.add` 逐元素操作的降维操作。

沿着指定维度 `axis` 减少输入 `input_tensor` 的维度：

- 除非 `keepdims` 为 true，否则对 `axis` 指定的维度，张量的秩都减少 1。
- 如果 `keepdims` 为 true，则缩减后的维度长度保持为 1。

如果 `axis` 为 `None`，则缩减所有维度，返回包含单个元素的张量。

> 与 `np.sum` 等价，不过 numpy 会将 uint8 和 int32 转换为 int64 类型，而 tensorflow 会保持原类型。

例如：

```python
>>> # x 的 shape 为 2 行 3 列 (2, 3)
>>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
>>> x.numpy()
array([[1, 1, 1],
       [1, 1, 1]])
>>> # 计算所有元素加和
>>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
>>> tf.reduce_sum(x).numpy()
6
>>> # 沿第 1 个轴计算
>>> # 结果是 [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
>>> tf.reduce_sum(x, 0).numpy()
array([2, 2, 2])
>>> # 沿第 2 个轴计算
>>> # 结果是 [1, 1] + [1, 1] + [1, 1] = [3, 3]
>>> tf.reduce_sum(x, 1).numpy()
array([3, 3])
>>> # 保持原始维度
>>> tf.reduce_sum(x, 1, keepdims=True).numpy()
array([[3],
       [3]])
>>> # 同时计算两个维度
>>> # 结果为：1 + 1 + 1 + 1 + 1 + 1 = 6
>>> # 或者，先计算行，再计算余下数组
>>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
>>> # 2 + 2 + 2 = 6
>>> tf.reduce_sum(x, [0, 1]).numpy()
6
```

## 参数

|参数|说明|
|---|---|
|input_tensor|待计算张量，必须为数值类型|
|axis|待计算维度，`None` (默认)表示消减所有维度，取值范围 `[-rank(input_tensor),rank(input_tensor)]`|
|keepdims|true 表示保持消减的维度为 1|
|name|（可选）操作名称|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
