# 深度学习计算

## GPU

2025-11-26⭐

本节讨论如何利用 GPU 的计算性能。首先是使用单个GPU，然后是如何使用多个GPU和多个服务器（具有多个GPU）。

具体来说，我们将讨论如何使用单个NVIDIA GPU进行计算。首先，确保至少安装了一个NVIDIA GPU。然后，下载[NVIDIA驱动和CUDA](https://developer.nvidia.com/cuda-downloads)并按照提示设置适当的路径。当这些准备工作完成，就可以使用`nvidia-smi`命令来查看显卡信息。

你可能已经注意到 `DJL` 的 `NDArray` 看起来与 NumPy 的 `ndarray` 非常类似。但有一些关键区别，其中之一是 DJL 支持不同的硬件设备。

在 `DJL` 中，每个数组都有一个设备（`Device`）。当我们跨多个服务器部署作业时，有的机器有 GPU, 有的没有，这使事情会变得棘手。默认情况下，`DJL` 会检测软硬件环境，自动选择高性能的运算设备来提高计算效率。

要运行此部分中的程序，至少需要两个GPU。注意，对于大多数桌面计算机来说，这可能是奢侈的，但在云中很容易获得，例如，通过使用AWS EC2的多GPU实例。本节几乎所有的其他部分都不需要多个GPU。本节只是为了说明数据如何在不同的设备之间传递。

### 计算设备

我们可以指定用于存储和计算的设备，如 CPU 和 GPU.

CPU 和 GPU 可以用 `Device.cpu()` 和 `Device.gpu()` 表示。需要注意的是，`Device.gpu(1)` 只代表一个卡和相应的显存。如果有多个GPU，我们使用`gpu(i)`表示第i块GPU（i从0开始）。另外，`gpu(0)`和`gpu()`是等价的。

```java
System.out.println(Device.cpu());
System.out.println(Device.gpu());
System.out.println(Device.gpu(1));
```

```
cpu()
gpu(0)
gpu(1)
```

我们可以查询可用gpu的数量。

```java

```

## 参考

- https://d2l-zh.djl.ai/chapter_deep-learning-computation/index.html