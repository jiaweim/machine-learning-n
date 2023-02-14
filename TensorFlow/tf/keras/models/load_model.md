# tf.keras.models.load_model

Last updated: 2022-10-09, 15:21
****

## 简介

```python
tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)
```

加载通过 `model.save()` 保存的模型。

例如：

```python
>>> model = tf.keras.Sequential([
...     tf.keras.layers.Dense(5, input_shape=(3,)),
...     tf.keras.layers.Softmax()])
>>> model.save('/tmp/model')
>>> loaded_model = tf.keras.models.load_model('/tmp/model')
>>> x = tf.random.uniform((10, 3))
>>> assert np.allclose(model.predict(x), loaded_model.predict(x))
```

请注意，加载后模型权重可能具有不同的 scope 名称。scope 名称包括 model/layer 名称，如 `"dense_1/kernel:0"`。建议使用 layer 属性访问特定变量，如 `model.get_layer("dense_1").kernel`。

## 参数

|参数|说明|
|---|---|
|filepath|以下之一：1. String 或 `pathlib.Path` 对象，SavedModel 路径；2. 模型所在的 `h5py.File` 对象|
|custom_objects|（可选）名称（string）到自定义类或函数（反序列化时要用的）的 dict|
|compile|Boolean, 是否在加载后编译模型|
|options|(可选) [tf.saved_model.LoadOptions](https://tensorflow.google.cn/api_docs/python/tf/saved_model/LoadOptions) 对象，指定加载 SavedModel 的选项|

返回 Keras 模型实例。如果原模型已编译，并且保存了 optimizer，则返回编译的模型。否则返回未编译模型。在返回未编译模型时，但设置 `compile=True`，将显示警告信息。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/keras/models/load_model
