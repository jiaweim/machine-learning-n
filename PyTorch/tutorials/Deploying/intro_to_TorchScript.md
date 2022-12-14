# TorchScript 简介

## 概述

TorchScript 是 PyTorch 模型（`nn.module` 的子类）的中间表示，可以在 C++ 等高性能环境中运行。

下面介绍：

1. PyTorch 模型创建基础，包括：
   - Modules
   - 定义 `forward` 函数
   - 组合 Module

2. 将 PyTorch 模块转换为 TorchScript 的方法，TorchScript 是 PyTorch 的高性能部署工具
   - 跟踪现有模块
   - 使用脚本直接编译模块
   - 组合这两种方法
   - 保存和接在 TorchScript 模块

```python
>>> import torch
>>> print(torch.__version__)
1.10.2+cu111
```

## PyTorch 模型创建基础



## 参考

- https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
