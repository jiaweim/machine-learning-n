# tf.compat.v1.GraphKeys

***

## 简介

包含 graph 中集合的标准名称。

标准库使用各种众所周知的名称来收集和检索与 graph 相关的值。例如，`tf.Optimizer` 子类默认优化 `tf.GraphKeys.TRAINABLE_VARIABLES` 中收集的变量，当然也可以显式传递待优化变量列表。

定义了如何标准集合名称：

- `GLOBAL_VARIABLES`：在分布式环境中共享的变量对象默认集合。详情参考 [tf.compat.v1.global_variables](https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables)。一般所有 `TRAINABLE_VARIABLES` 变量在 `MODEL_VARIABLES` 中，而所有 `MODEL_VARIABLEES` 变量又在 `GLOBAL_VARIABLES` 中。
- `LOCAL_VARIABLES`：每台机器的 local 变量。一般用来存储临时变量，如用来计数。`tf.contrib.framework.local_variable` 函数将变量添加到该集合。
- `MODEL_VARIABLES`：模型前向传播（inference）使用的变量。`tf.contrib.framework.model_variable` 函数将变量添加到该集合。
- `TRAINABLE_VARIABLES`：由优化器（optimizer）训练的变量。详情参考 [tf.compat.v1.trainable_variables](https://www.tensorflow.org/api_docs/python/tf/compat/v1/trainable_variables)
- `SUMMARIES`：

## 参考

- https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphKeys
