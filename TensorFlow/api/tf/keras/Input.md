# tf.keras.Input

2022-03-09, 22:23
***

## 简介

```python
tf.keras.Input(
    shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
    ragged=None, type_spec=None, **kwargs
)
```

`Input()` 用于实例化 Keras 张量。

Keras 张量是类似符号张量的对象，并使用属性来增强，通过这些属性使得我们可仅通过输入和输出构建 Keras 模型。

例如，如果 `a`, `b` 和 `c` 是 Keras 张量，则可以如此定义模型：`model = Model(input=[a, b], output=c)`。

| 参数  | 说明  |
|---|---|
|   |   |
|shape|shape tuple (integers)，不包括 batch size。例如 `shape=(32,)` 表示期望的输入是批量的 32 维张量。该 tuple 的元素为 `None`，表示对应维度未知|
|batch_size|optional static batch size (integer).|
|name|layer 的可选名称，在模型中不允许重复，如果不提供，会自动生成。|
|dtype|The data type expected by the input, as a string (float32, float64, int32...)|
|sparse|A boolean specifying whether the placeholder to be created is sparse. Only one of 'ragged' and 'sparse' can be True. Note that, if sparse is False, sparse tensors can still be passed into the input - they will be densified with a default value of 0.|
|tensor|Optional existing tensor to wrap into the Input layer. If set, the layer will use the tf.TypeSpec of this tensor rather than creating a new placeholder tensor.
|ragged|A boolean specifying whether the placeholder to be created is ragged. Only one of 'ragged' and 'sparse' can be True. In this case, values of 'None' in the 'shape' argument represent ragged dimensions. For more information about RaggedTensors, see this guide.|
|type_spec|A tf.TypeSpec object to create the input placeholder from. When provided, all other args except name must be None.|
|**kwargs|deprecated arguments support. Supports batch_shape and batch_input_shape.|

## 示例

```python
# this is a logistic regression in Keras
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/Input
