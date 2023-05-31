# GlobalAveragePooling1D

- [GlobalAveragePooling1D](#globalaveragepooling1d)
  - [简介](#简介)
  - [参数](#参数)
  - [示例](#示例)
  - [参考](#参考)

Last updated: 2022-06-30, 10:41
@author Jiawei Mao
****

## 简介

时序数据的全局平均池化操作。

```python
tf.keras.layers.GlobalAveragePooling1D(
    data_format='channels_last', **kwargs
)
```

## 参数

|参数  |说明  |
|------|-----|
|data_format |`channels_last` (默认) 或 `channels_first`，输入维度排序。`channels_last` 对应输入 shape `(batch, steps, features)`, `channels_first` 对应输入 shape `(batch, features, steps)`         |
|keepdims |boolean, 是否保留时间维度。`False` (默认) 表示不保留时间维度，返回的张量 rank 减 1.`True` 则保留时间维度，长度为 1。该行为与 `tf.reduce_mean` 或 `np.mean` 一致|

|调用参数 |说明  |
|------|-----|
|inputs|3D 张量|
|mask|shape 为 `(batch_size, steps)` 的二元张量，表示是否屏蔽指定时间步（不做平均）|

输入 shape：

- 如果 `data_format='channels_last'`，输入 3D 张量 shape 为 `(batch_size, steps, features)`
- 如果 `If data_format='channels_first'`，输入 3D 张量 shape 为 `(batch_size, features, steps)`

输出 shape：

- 如果 `keepdims=False`，返回 `(batch_size, features)` 2D 张
- 如果 `keepdims=True`
  - 如果 `data_format='channels_last'`，返回 `(batch_size, 1, features)` 3D 张量
  - 如果 `data_format='channels_first'`，返回 `(batch_size, features, 1)` 3D 张量

## 示例

```python
>>> import tensorflow as tf
>>> input_shape = (2, 3, 4) # 序列是导数第二个 axis
>>> x = tf.random.normal(input_shape)
>>> x
<tf.Tensor: shape=(2, 3, 4), dtype=float32, numpy=
array([[[-0.9039546 ,  0.12750119, -1.1285012 , -0.79991406],
        [ 0.99187106, -0.3415825 , -0.08723515, -0.91464037],
        [ 0.37513864,  0.14899828,  0.00668311,  0.14803089]],

       [[-0.4324134 , -0.8456199 ,  0.0273356 ,  1.8411161 ],
        [-0.23405078, -0.6646851 ,  0.49210122, -0.23742162],
        [-1.2154015 ,  0.66094667, -0.30357808,  0.4437055 ]]],
      dtype=float32)>
>>> y = tf.keras.layers.GlobalAveragePooling1D()(x) # 对序列进行平均
>>> y
<tf.Tensor: shape=(2, 4), dtype=float32, numpy=
array([[ 0.1543517 , -0.02169435, -0.40301776, -0.52217454],
       [-0.6272886 , -0.2831194 ,  0.07195292,  0.6824667 ]],
      dtype=float32)>
>>> print(y.shape)
(2, 4)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
