# 深度学习

## 简介

深度学习算法，如卷积神经网络、transformer等，利用分层结构将输入数据转化为抽象表示。

相对传统机器学习算法，深度学习能自动从数据中发现有用的特征。

smile-core 模块提供 MLP 模块，而 smile-deep 为计算机视觉和大型语言模型（LLM）提供高级算法。此外，smile-deep 支持 GPU 设备。

## 示例

下面展示如何在 MNIST 数据集上训练模型。

- 调用 `Device.preferredDevice()` 获取 GPU 设备（如果有），否则返回 CPU 设备
- 可以调用工厂方法创建 `Device`，例如：`Device.GPU(0)`, `Device.MPS()`, `Device.CPU()`，然后将返回的设备设置为默认设备



## 参考

- https://haifengl.github.io/deep-learning.html