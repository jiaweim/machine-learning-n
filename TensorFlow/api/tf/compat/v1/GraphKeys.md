# tf.compat.v1.GraphKeys

## 简介

包含 graph 中集合的标准名称。

标准库使用各种众所周知的名称来收集和检索与 graph 相关的值。例如，`tf.Optimizer` 子类默认优化 `tf.GraphKeys.TRAINABLE_VARIABLES` 中收集的变量，当然也可以显式传递待变量列表。

定义了如何标准集合名称：

- `GLOBAL_VARIABLES`：

## 参考

- https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphKeys
