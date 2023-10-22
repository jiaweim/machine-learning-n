# TensorFlow 1.x vs TensorFlow 2

## 简介

TensorFlow 2 与 TF1.x 的编程范式完全不同。

下面从行为和 API 两个角度描述 TF1.x 和 TF2 的基本区别，以及如何跟你举这些区别进行迁移。

## 主要变化总结

从根本上讲，TF1.x 和 TF2 围绕执行、变量、控制流、tensor shape 和 tensor equality comparison 使用了一套不同的运行时行为（eager in TF2）。要兼容 TF2，你的代码必须与完整的 TF2 行为兼容。


## 参考

- https://www.tensorflow.org/guide/migrate/tf1_vs_tf2
