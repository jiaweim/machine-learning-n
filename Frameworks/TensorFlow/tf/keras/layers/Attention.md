# tf.keras.layers.Attention

***

## 简介

```python
tf.keras.layers.Attention(
    use_scale=False, score_mode='dot', **kwargs
)
```

Dot-product attention layer, 即 Luong-style attention.

输入包括 `query` tensor（`[batch_size, Tq, dim]`），`value` tensor（`[batch_size, Tv, dim]`），`key` tensor（`[batch_size, Tv, dim]`）。计算步骤如下：

1. 根据 `query-key` dot product 计算 attention score (`[batch_size, Tq, Tv]`): `scores = tf.matmul(query, key, transpose_b=True)`
2. 使用 attention score 计算分布（`[batch_size, Tq, Tv]`）：`distribution = tf.nn.softmax(scores)`
3. 使用 `distribution` 创建 `value` 的线性组合（[batch_size, Tq, dim]）：`return tf.matmul(distribution, value)`

## 参数

|参数|说明|
|---|---|
|use_scale|`True` 表示用一个标量来缩放 attention score|
|dropout|0 到 1 的 float。舍弃 attention score 单元比例，默认 0.0|
|score_mode|计算 attention score 的函数，可选 `{"dot", "concat"}`。`"dot"` 指 query 和 key 向量之间的点乘；`"concat"` 指 query 和 key 串联后的双曲正切|

## 调用参数

|参数|说明|
|---|---|
|inputs|如下 tensor list:<li> query: Query tensor `[batch_size, Tq, dim]`<li>value: Value tensor `[batch_size, Tv, dim]`<li>key: 可选的 key tensor `[batch_size, Tv, dim]`。如果不提供，`key` 和 `value` 都使用 `value`，这是最常见的情形|
|mask|如下 tensor list:
query_mask: A boolean mask Tensor of shape [batch_size, Tq]. If given, the output will be zero at the positions where mask==False.
value_mask: A boolean mask Tensor of shape [batch_size, Tv]. If given, will apply the mask such that values at positions where mask==False do not contribute to the result.
return_attention_scores	bool, it True, returns the attention scores (after masking and softmax) as an additional output argument.
training	Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout).
use_causal_mask	Boolean. Set to True for decoder self-attention. Adds a mask such that position i cannot attend to positions j > i. This prevents the flow of information from the future towards the past. Defaults to False.

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Attention
