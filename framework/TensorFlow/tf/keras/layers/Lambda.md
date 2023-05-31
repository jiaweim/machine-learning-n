# tf.keras.layers.Lambda

Last updated: 2023-02-18, 16:23
****

## 简介

```python
tf.keras.layers.Lambda(
    function, output_shape=None, mask=None, arguments=None, **kwargs
)
```

将任意表达式包装为 `Layer` 对象。

`Lambda` 层时为了将任意表达式包装为 `Layer`，便于 `Sequential` 和 `Functional` API 使用。`Lambda` 层适合于简单操作或快速试验。对更复杂的情况，建议[继承 tf.keras.layers.Layer](https://www.tensorflow.org/guide/keras/custom_layers_and_models)。

> **WARNING**
> 继承 `tf.keras.layers.Layer` 而不是使用 `Lambda` 层的主要用于，是便于保存和检查模型。`Lambda` 层通过序列化 Python 字节码来保存，所以基本上不可移植，一般只能在保存它们的环境中加载。子类 API 通过覆盖 `get_config` 方法实现可移植。而且使用子类 API 层的模型更容易可视化和检查。

## 示例

```python
# 计算平方的 layer
model.add(Lambda(lambda x: x ** 2))
```

```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

model.add(Lambda(antirectifier))
```

## Variables

虽然可以在 Lambda 层中使用 Varibale，但是不推荐，很容易出错。例如：

```python
scale = tf.Variable(1.)
scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
```

因为 `scale_layer` 不直接跟踪 `scale` 变量，所以它不会出现在 `scale_layer.trainable_weights`，因此如果在模型中使用 `scale_layer`，`scale` 变量不会被训练。

实现子类 Layer 是更好的方式：

```python
  class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ScaleLayer, self).__init__()
      self.scale = tf.Variable(1.)

    def call(self, inputs):
      return inputs * self.scale
```

一般来说，`Lambda` 层可以进行简单的无状态计算，一旦涉及更复杂的事情都应该使用子类 Layer 来代替。

## 参数



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda
