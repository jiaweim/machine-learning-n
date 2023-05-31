# TorchVision 对象检测和 Finetuning

- [TorchVision 对象检测和 Finetuning](#torchvision-对象检测和-finetuning)
  - [简介](#简介)
  - [定义数据集](#定义数据集)
  - [参考](#参考)

***

## 简介

下面介绍使用 [Penn-Fudan 数据集](https://www.cis.upenn.edu/~jshi/ped_html/) 微调预训练模型 [Mask R-CNN](https://arxiv.org/abs/1703.06870)。Penn-Fudan 数据集包含 170 张图像，共 345 个行人的图像，下面使用该数据集介绍 torchvision 的功能。

## 定义数据集

数据集要继承 `torch.utils.data.Dataset` 类，实现 `__len__` 和 `__getitem__` 方法。

唯一要注意的是 `__getitem__` 应该返回：

- image: size 为 `(H, W)` 的 PIL 图像
- target: 包含如下字段的 dict
  - `boxes (FloatTensor[N, 4])`：`N` 个 `[x0, y0, x1, y1]` 格式的边界框坐标，数值从 0 到 W 以及 0 到 H；
  - `labels (Int64Tensor[N])`：每个边框的标签，`0` 表示背景；
  - `image_id (Int64Tensor[1])`：图像识别符，在数据集中 unique，在 evaluation 期间使用；
  - `area (Tensor[N])`：边界框的面积。使用 COCO 指标进行评估时使用，用来区分小、中、大边界框的指标打分；
  - `iscrowd (UInt8Tensor[N])`：评估时忽略 `iscrowd=True` 的样本；
  - （可选）`masks (UInt8Tensor[N, H, W])`：每个对象的分隔 mask；
  - （可选）`keypoints (FloatTensor[N, K, 3])`：图像中的 N 个对象的每一个，都包含 `[x, y, visibility]` 格式的


## 参考

- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
