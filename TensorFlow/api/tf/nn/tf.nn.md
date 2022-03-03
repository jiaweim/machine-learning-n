# tf.nn

2022-03-03, 01:10
***

## 简介

基础神经网络操作。

## padding

部分神经网络操作，如 `tf.nn.conv2d` 和 `tf.nn.max_pool2d` 包含 `padding` 参数，该参数用于控制在操作前对输入的填充。通过在张量的前后插入值（一般为零）进行填充。`padding` 参数可选项包括 `'VALID'` 和 `'SAME'`，`'VALID'` 表示不填充，而 `'SAME'` 的填充规则下面会详细说明。某些操作允许通过 list 显式指定每个维度的填充值。

在卷积操作中，输入用零填充。对池化操作，填充的输入值被忽略。例如，在最大池化中，sliding window 忽略填充值，相当于填充值为 `-infinity`。

### VALID 填充

`padding='VALID'` 表示无填充。这通常会导致输出 size 小于输入 size，即使 stride 为 1。对 2D，输出 size 计算方法如下：

```python
out_height = ceil((in_height - filter_height + 1) / stride_height)
out_width  = ceil((in_width - filter_width + 1) / stride_width)
```

1D 和 3D 情况类似。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/nn
