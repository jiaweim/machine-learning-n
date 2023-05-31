# tf.keras.models.save_model

Last updated: 2022-10-09, 15:55
****

## 简介

```python
tf.keras.models.save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True
)
```

将模型保存为 TF SavedModel 或 HDF5 文件。

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

注意，`model.save()` 是 `tf.keras.models.save_model()` 的别名。

SavedModel 和 HF5 文件包含：

- 模型架构
- 模型权重
- 模型 optimizer 状态

因此，模型可以重新实例化为完全相同的状态，无需任何定义或训练代码。

注意，模型加载后权重的 scope 名称可能不同。scope 名称包括 model/layer 名称，例如 `"dense_1/kernel:0"`。建议使用 layer 属性访问特定变量，例如 `model.get_layer("dense_1").kernel`。

**SavedModel 序列化格式**

Keras SavedModel 使用 `tf.saved_model.save` 保存模型和所有附加到模型的可追踪对象（如 layers 和变量）。模型 config, weights 和 optimizer 保存在 SavedModel 中。此外，对附加到模型的所有 Keras layers，SavedModel 保存：

- config 和 metadata，即 name, dtype, trainable status
- 记录所有 call 和 loss 函数，保存为 TF subgraph。

记录的函数使得 SavedModel 格式无需原始类定义可以保存和加载自定义 layers。

可以禁用 `save_traces` 选项选择不保存记录的函数。这将减少保存模型所需的时间以及输出的 SavedModel 占用的磁盘空间。如果启用该选项，则必须在加载模型时提供所需的自定义类。

## 参数

|参数|说明|
|---|---|
|model|要保存的 Keras 模型实例|
|filepath|下列选项之一：1. String 或 `pathlib.Path` 对象，保存模型的路径；2. 保存模型的 `h5py.File` 对象|
|overwrite|是否覆盖目标位置的已有模型，或提示用户|
|include_optimizer|是否同时保存 optimizer 状态|
|save_format|'tf' 或 'h5'，将模型保存为 TF SavedModel 还是 HDF5 格式。TF 2.x 中默认为 'tf'，TF 1.x 中默认为 'h5'|
|signatures|与 SavedModel 一同保存的签名。只适用于 "tf" 格式。详情请参考 [tf.saved_model.save](https://tensorflow.google.cn/api_docs/python/tf/saved_model/save)|
|options|(仅适用于 SavedModel 格式) [tf.saved_model.SaveOptions](https://tensorflow.google.cn/api_docs/python/tf/saved_model/SaveOptions) 对象，指定保存为 SavedModel 的选项|
|save_traces|(仅适用于 SavedModel 格式) 启用后，SavedModel 将保存每个 layer 的函数 traces。可以禁用该选项，这样就只保存每层的 config。**默认** True。禁用此选项可以减少序列化时间和文件大小，但要求所有自定义 layer/model 实现 `get_config()` 方法|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
- https://tensorflow.google.cn/api_docs/python/tf/keras/models/save_model
