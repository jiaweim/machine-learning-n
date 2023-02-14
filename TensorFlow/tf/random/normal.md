# tf.random.normal

2022-02-23, 15:15
****

## 简介

```python
tf.random.normal(
    shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None
)
```

从正态分布生成随机数。

## 参数

|参数|说明|
|---|---|
|shape|A 1-D integer Tensor or Python array. The shape of the output tensor.|
|mean|A Tensor or Python value of type dtype, broadcastable with stddev. The mean of the normal distribution.|
|stddev|A Tensor or Python value of type dtype, broadcastable with mean. The standard deviation of the normal distribution.|
|dtype|The float type of the output: float16, bfloat16, float32, float64. Defaults to float32.|
|seed|A Python integer. Used to create a random seed for the distribution. See tf.random.set_seed for behavior.|
|name|A name for the operation (optional).|

## 示例

每次生成一组新的随机数：

```python
>>> tf.random.set_seed(5);
>>> tf.random.normal([4], 0, 1, tf.float32)
tf.Tensor([-0.18030666 -0.95028627 -0.03964049 -0.7425406 ], shape=(4,), dtype=float32)
```

输出可重现的结构：

```python
>>> tf.random.set_seed(5);
>>> tf.random.normal([2,2], 0, 1, tf.float32, seed=1)
tf.Tensor(
[[-1.3768897  -0.01258316]
 [-0.169515    1.0824056 ]], shape=(2, 2), dtype=float32)
```

这里同时设置全局 seed 和操作 seed，以保证生成的随机序列可重现。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/random/normal
