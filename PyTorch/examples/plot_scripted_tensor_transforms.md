# 张量变换和 JIT

## 简介

下面介绍支持的各种图像张量变换功能。特别是如何在 GPU 上执行图像变换，以及如何使用 JIT 编译脚本。

在 v0.8.0 之前，torchvision 的变换以 PIL 为中心，存在许多限制。现在，变换同时兼容张量和 PIL，实现了如下新功能：

- 多通道张量图像的变换（3-4 通道）
- 

## 参考

- https://pytorch.org/vision/main/auto_examples/plot_scripted_tensor_transforms.html
