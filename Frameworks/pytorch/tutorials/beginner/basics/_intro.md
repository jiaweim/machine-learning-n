# 基础知识

@since 2026-08-17⭐
@author Jiawei Mao
***

大多数机器学习流程包含数据处理、构建模型、优化模型参数以及保存模型这几个步骤。下面介绍一套基于 PyTorch 的完整机器学习工作流程。

我们将使用 FashionMNIST 数据集训练一个神经网络模型，该模型能够判断输入图片属于以下哪一类物品：T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag 以及 Ankle boot。

阅读本教程需要具备基础的 Python 编程能力，并了解深度学习相关基础概念。

如果你熟悉其他深度学习框架，可以先阅读 0. 快速入门，快速上手 PyTorch 的应用程序接口。

如果你是深度学习框架新手，请直接从分步教程的第一节开始学习：1. 张量 (Tensors)。

- [快速入门](./quickstart_tutorials.md)
- 张量
- 数据集与数据加载器
- 数据变换
- 构建模型
- 自动微分
- 优化循环（训练迭代）
- 模型的保存、加载与使用

## 参考

- https://docs.pytorch.org/tutorials/beginner/basics/intro.html