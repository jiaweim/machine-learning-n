# 快速入门

- [快速入门](#快速入门)
  - [简介](#简介)
  - [数据处理](#数据处理)
  - [参考](#参考)

2021-12-15, 14:00
***

## 简介

这部分介绍机器学习中常见任务的 API。

## 数据处理

PyTorch 包含两个处理数据的基本类：

- `torch.utils.data.Dataset`，保存样本及其对应的标签；
- `torch.utils.data.DataLoader`，`DataLoader` 包装 `Dataset`，提供可迭代对象。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
