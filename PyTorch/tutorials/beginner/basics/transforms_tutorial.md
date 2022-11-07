# 转换

- [转换](#转换)
  - [简介](#简介)
  - [ToTensor](#totensor)
  - [Lambda Transform](#lambda-transform)
  - [参考](#参考)

Last updated: 2022-11-07, 18:53
****

## 简介

数据并不总是以训练机器学习算法所需的形式出现。因此需要对数据进行**转换**（transform），使其适合于训练模型。

所有 TorchVision 数据集有两个参数：`transform` 用于转换特征，`target_transform` 用于转换标签，它们是包含转换逻辑的可调用对象。[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) 模块提供了几个开箱即用的常用转换。

FashionMNIST 数据集的图像是 PIL 图像格式，标签为整数。为了训练，需要将特征转换为归一化张量，将标签转换为 one-hot 编码张量。分别使用 `ToTensor` 和 `Lambda` 进行转换。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="D:\data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

## ToTensor

[ToTensor](https://pytorch.org/vision/stable/transforms.html) 将 PIL 图像或 NumPy `ndarray` 转换为 `FloatTensor`，并将图像的像素值缩放到 [0., 1.] 之间。

## Lambda Transform

Lambda transform 应用任意用户定义的 lambda 函数。上面定义一个将整数转换为 one-hot 编码张量的函数。它首先创建大小为 10 的全 0 张量，然后调用 `scatter_` 将标签 `y` 对应索引复制为 `value=1`.

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
