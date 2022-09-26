# tf.keras.layers.ActivityRegularization

Last updated: 2022-09-26, 16:54
****

## 简介

```python
tf.keras.layers.ActivityRegularization(
    l1=0.0, l2=0.0, **kwargs
)
```

基于输入激活值（activity）更新 cost 函数。

## 参数

|参数|说明|
|---|---|
|l1|L1 正则化因子 (positive float)|
|l2|L2 正则化因子 (positive float)|

**输入 shape**

不限制。当作为模型的第一层使用，使用关键字参数 `input_shape` 指定 shape。

**输出 shape**

同输入。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/keras/layers/ActivityRegularization
