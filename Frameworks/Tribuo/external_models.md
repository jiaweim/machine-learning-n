# 使用外部模型

## 简介

Tribuo 可以加在第三方系统中训练的模型，并将其与原生 Tribuo 模型一起部署。Tribuo 4.1+ 支持 XGBoost、TensorFlow  frozen graphs 和 saved modesl，以及以 ONNX 格式导出的模型。特别是对 ONNX 的支持，许多库可以导出为 ONNX 格式，如 scikit-learn, pytorch, TensorFlow 等。onnx 支持的完整列表可以参考 [ONNX 官网](https://onnx.ai/)。Tribuo 的 ONNX 支持通过 [ONNX Runtime](https://onnxruntime.ai/) 实现，其 Java 接口由 Oracle Labs 贡献。Tribuo 4.2 添加了将模型导出为 ONNX 格式的功能，并且可以使用 ONNX Runtime 接口重新加载回 Tribuo。

下面介绍如何加载在 XGBoost, scikit-learn 和 pytorch 中训练的模型（全部基于 MNIST），并将它们部署到 Tribuo。在 [TensorFlow 教程](./tensorflow_tribuo.md) 中专门讨论了如何使用 TensorFlow 模型，因为 TensorFlow 本身就很复杂。注意，这些模型都依赖于 native 库，支持 x86_64 Windows, Linux 和 macOS。

## 参考

- https://tribuo.org/learn/4.3/tutorials/external-models-tribuo-v4.html