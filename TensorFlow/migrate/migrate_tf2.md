# TF1.x 到 TF2 迁移概述

- [TF1.x 到 TF2 迁移概述](#tf1x-到-tf2-迁移概述)
  - [简介](#简介)
  - [TF2 迁移过程](#tf2-迁移过程)
  - [参考](#参考)

***

## 简介

TF2 在很多方面与 TF1.x 有根本的不同。不过依然可以按如下方式在 TF2 中运行 TF1.x 代码（[contrib 除外](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)）：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

这种方式抛弃了 TF2 的优点，无法运行 TF2 API 和为 TF2 编写的代码。如果未使用任何 TF2 特性，则没问题。[TensorFlow 1.x vs TensorFlow 2](https://www.tensorflow.org/guide/migrate/tf1_vs_tf2) 中有 TF2 和 TF1.x 差别的详细介绍。

本指南概述了将 TF1.x 迁移到 TF2 的过程。从而可以使用 TF2 新的功能和更新，使代码更简单、性能更好，且更易于维护。

如果使用 tf.keras API，并且只使用 `model.fit` 进行训练，则代码基本与 TF2 兼容，只需要注意以下几点：

- TF2 中 Keras optimizer 的[默认学习率](https://www.tensorflow.org/guide/effective_tf2#optimizer_defaults)不同
- TF2 更改了一些 [metric 的 "name"](https://www.tensorflow.org/guide/effective_tf2#keras_metric_names)

## TF2 迁移过程

在迁移前，要充分了解 TF1.x 和 TF2 的差比。



## 参考

- https://www.tensorflow.org/guide/migrate/migrate_tf2
