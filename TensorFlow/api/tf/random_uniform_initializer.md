# random_uniform_initializer

Last updated: 2022-07-07, 13:00
@author Jiawei Mao
****

## 简介

```python
tf.random_uniform_initializer(
    minval=-0.05, maxval=0.05, seed=None
)
```

生成均匀分布张量的初始化器。

初始化器（initializer）可用于提前指定初始化策略，不需要知道待初始化张量的 shape 和 dtype。

## 参数

|参数|说明|
|---|---|
|minval|Python 标量或标量张量，指定随机值范围下限 (inclusive).|
|maxval|Python 标量或标量张量，指定随机值范围上限 (exclusive).|
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
>>> def make_variables(k, initializer): # 使用初始化器生成张量
...     return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
...             tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
>>> v1, v2 = make_variables(3, tf.ones_initializer()) # 用 1 初始化
>>> v1
<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>
>>> v2
<tf.Variable 'Variable:0' shape=(3, 3) dtype=float32, numpy=
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]], dtype=float32)>
>>> # 用均匀分布初始化
>>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.)) 
(<tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([-0.8790145 ,  0.3770554 , -0.6247859 ,  0.18160462], dtype=float32)>,
 <tf.Variable 'Variable:0' shape=(4, 4) dtype=float32, numpy=
 array([[ 0.18614888, -0.6593411 , -0.30814958, -0.66844106],
        [-0.8119247 ,  0.1132431 , -0.35387397, -0.33532   ],
        [-0.87777305, -0.5755458 ,  0.6754377 ,  0.9387469 ],
        [ 0.07272649, -0.7860918 , -0.80084896, -0.6775    ]],
       dtype=float32)>)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer
