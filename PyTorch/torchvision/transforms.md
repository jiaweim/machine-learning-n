# 图像变换和增强

- [图像变换和增强](#图像变换和增强)
  - [简介](#简介)
  - [Compositions of transforms](#compositions-of-transforms)
  - [Transforms on Tensor only](#transforms-on-tensor-only)
  - [总结](#总结)
    - [Compose](#compose)
    - [Normalize](#normalize)
  - [参考](#参考)

***

## 简介

变换（Transform）是 `torchvision.transforms` 模块提供的常见的图像变换功能。多个变换可以使用 `Compose` 串在一起。大多数变换类都有一个等价的函数：变换函数可以实现对变换的细粒度控制，对构建复杂的变换管线非常有用。

大多数变换同时接受 PIL 图像和图像张量，少数只支持 PIL 或图像张量。可以用转换变换实现 PIL 图像和张量之间的转换。

接受图像张量的变换也接受批量图像张量。图像张量 shape 为 $(C, H, W)$，其中 `C` 指通道数，`H` 和 `W` 是图像的高度和宽度。批量图像张量的 shape 为 $(B, C, H, W)$，B 表示 batch 数。

图像张量的数值范围由张量类型 `dtype` 隐式定义。浮点类型的图像张量值范围应为 `[0,1)`。整数类型的图像张量的值应为 `[0, MAX_DTYPE]`，`MAX_DTYPE` 表示该数据类型可以表示的最大值。

随机变换对给定批次的图像应用相同的变换，但是在调用之间产生不同的变换。要实现可重复的变化，可以使用函数变换。

> **WARNING**
> 从 v0.8.0 开始，所有随机变换都使用 torch 默认随机生成器对随机参数进行采样。该变化破坏了向后兼容，用户应按如下方式设置随机状态：
> ```python
> # Previous versions
> # import random
> # random.seed(12)
> # Now
> import torch
> torch.manual_seed(17)
> ```
> torch 随机生成器和 Python 随机生成器即使 seed 相同，生成结果也不同。

## Compositions of transforms

## Transforms on Tensor only



## 总结

### Compose

```python
torchvision.transforms.Compose(transforms)
```

将多个变换组合在一起。该变换不支持

### Normalize

```python
torchvision.transforms.Normalize(mean, std, inplace=False)
```

根据均值和方差归一化张量图像。该变换不支持 PIL 图像。对 `n` 个通道的均值 `(mean[1],...,mean[n])` 和标准差 `(std[1],..,std[n])`，该变换对每个通道依次进行归一化，即 `output[channel] = (input[channel] - mean[channel]) / std[channel]`。

> **NOTE**：该变换不是原地操作，即不改变输入张量。

## 参考

- https://pytorch.org/vision/stable/transforms.html
