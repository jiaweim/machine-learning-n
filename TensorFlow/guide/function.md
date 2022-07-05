# tf.function 性能

## 简介

TF 2 默认启用即时执行（eager execution）。用户接口直观灵活，运行一次性操作更容易、快捷，但这可能以牺牲性能和可部署性为代价。

可以使用 `tf.function` 