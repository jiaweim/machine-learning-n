# tf.keras.layers.Attention

## 简介

```python
tf.keras.layers.Attention(
    use_scale=False, score_mode='dot', **kwargs
)
```

Dot-product attention layer, 即 Luong-style attention.

输入包括 shape 为 `[batch_size, Tq, dim]` 的 `query` tensor，shape 为 `[batch_size, Tv, dim]` 的 `value` tensor，shape 为 `[batch_size, Tv, dim]` 的 `key` tensor。计算步骤如下：

1. 

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Attention
