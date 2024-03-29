# tf.keras.Input

Last updated: 2022-10-27, 15:35
****

## 简介

```python
tf.keras.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=None,
    tensor=None,
    ragged=None,
    type_spec=None,
    **kwargs
)
```

`Input()` 用于实例化 Keras 张量。

Keras tensor 是类符号张量的对象，并使用属性进行增强，通过这些属性只需要输入和输出就能构建 Keras 模型。

例如，如果 `a`, `b` 和 `c` 是 Keras 张量，则可以如此定义模型：`model = Model(input=[a, b], output=c)`。

## 参数

| 参数  | 说明  |
|---|---|
|shape|shape tuple (integers)，不包括 batch size。例如 `shape=(32,)` 表示期望输入是批量的 32 维向量。该 tuple 的元素为 `None` 时，表示对应维度未知|
|batch_size|（可选）batch 大小（int）|
|name|（可选）layer 名称，在模型中不允许重复，如果不提供，会自动生成|
|dtype|输入类型，string (`float32`, `float64`, `int32`...)|
|sparse|boolean，指定要创建的占位符是否稀疏。`ragged` 和 `sparse` 只能有一个为 True。如果 `sparse=False`，仍可以传入稀疏张量，默认以 0 填充为密集 tensor|
|tensor|（可选）包装到 `Input` 层的现有张量。设置后，该 layer 使用现有张量的 `tf.TypeSpec`，而不用创建一个新的占位张量|
|ragged|boolean，指定要创建的占位符是否为参差张量。`ragged` 和 `sparse` 只能有一个为 True。如果 `ragged=True`，`shape` 参数中的 `None` 表示 ragged 维度|
|type_spec|创建输入占位符的 `tf.TypeSpec` 对象。如果提供该参数，除 `name` 外的其它参数都必须为 `None|
|**kwargs|不推荐的参数。支持 `batch_shape` 和 `batch_input_shape`|

返回一个张量。

## 示例

- Keras 实现逻辑回归

```python
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

注意，即使启用了 eager 执行，`Input` 也会生成一个符号张量对象（即占位符）。该符号张量可用作以张量为输入的低级 TF 操作，例如：

```python
x = Input(shape=(32,))
y = tf.square(x)  # 该操作被当作一个 layer
model = Model(x, y)
```

> 此行为不适合于高级 TF API，例如由 `tf.GradientTape` 直接监控的控制流。

但是，生成的模型不会跟踪任何 TF 操作的输入变量。所有变量的使用都必须在 Keras layer 中发生，以确保通过模型的 weights 跟踪。

Keras `Input` 也可以使用 `tf.TypeSpec` 创建占位符，例如：

```python
x = Input(type_spec=tf.RaggedTensorSpec(shape=[None, None],
                                        dtype=tf.float32, ragged_rank=1))
y = x.values
model = Model(x, y)
```

当传递 `tf.TypeSpec` 时，它必须表示整个批量的签名，而不仅仅是一个样本。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/Input
