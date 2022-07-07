# random_normal_initializer

Last updated: 2022-07-07, 13:27
@author Jiawei Mao
****

## 简介

```python
tf.random_normal_initializer(
    mean=0.0, stddev=0.05, seed=None
)
```

生成正态分布张量的初始化器。

初始化器（initializer）可用于提前指定初始化策略，不需要知道待初始化张量的 shape 和 dtype。

## 参数

|参数|说明|
|---|---|
|mean|Python 标量或标量张量，指定正态分布均值|
|stddev|Python 标量或标量张量，指定正态分布标准差|
|seed|Python 整数，用来创建随机 seed|

## 方法

### from_config

```python
@classmethod
from_config(
    config
)
```

使用指定配置 dict 实例化初始化器。例如：

```python
initializer = RandomUniform(-1, 1)
config = initializer.get_config()
initializer = RandomUniform.from_config(config)
```

- config

包含配置信息的 Python dict，一般是 `get_config` 的返回值。

### get_config

```python
get_config()
```

以 JSON 序列化 dict 格式返回初始化器的配置。

### `__call__`

```python
__call__(
    shape,
    dtype=tf.dtypes.float32,
    **kwargs
)
```

返回以该初始化器初始化的张量对象。

- shape

张量 shape.

- dtype

张量类型，支持浮点和整数类型。

- `**kwargs`

其它关键字参数。

## 示例

```python
>>> def make_variables(k, initializer):
...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
>>> v1, v2 = make_variables(3, tf.random_normal_initializer(mean=1., stddev=2.))
>>> v1
<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([3.5688255 , 0.15198952, 2.6019406 ], dtype=float32)>
>>> v2
<tf.Variable 'Variable:0' shape=(3, 3) dtype=float32, numpy=
array([[ 0.1959961 ,  0.11585981,  1.5133286 ],
       [-0.39734733,  0.87649715, -0.8246238 ],
       [ 2.0587773 , -0.12920928,  1.1217571 ]], dtype=float32)>
>>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
(<tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([ 0.84345055, -0.52507925, -0.6453583 ,  0.3229463 ], dtype=float32)>,
 <tf.Variable 'Variable:0' shape=(4, 4) dtype=float32, numpy=
 array([[-0.78820896, -0.74353695,  0.09158039, -0.26500225],
        [ 0.6582954 , -0.76843715,  0.07229853, -0.4713936 ],
        [ 0.24700403, -0.66098857,  0.5064628 ,  0.3843336 ],
        [ 0.86569405,  0.3499627 , -0.53868985, -0.8365185 ]],
       dtype=float32)>)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/random_normal_initializer
