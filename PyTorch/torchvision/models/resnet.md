# ResNet

- [ResNet](#resnet)
  - [简介](#简介)
  - [Model builders](#model-builders)
  - [resnet50](#resnet50)
  - [参考](#参考)

***

## 简介

ResNet 模型基于论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 实现。

> **NOTE**：Torchvision 将下采样的 stride 设置为第二个 3x3 卷积，而原始文献将其设置为第一个 1x1 卷积。该变体提高了准确性，称为 [ResNet V1.5](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch).

## Model builders

以下 model builder 可用来实例化 ResNet 模型。所有 model builder 内部都依赖于 `torchvision.models.resnet.ResNet` 类。详情可参考[源码](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)。

## resnet50

```python
torchvision.models.resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) → ResNet
```

参数：

- **weights** (`ResNet50_Weights`, optional)：要使用的预训练权重。默认不适用。
- **progress** (`bool`, optional)：True 表示显示下载的进度条。默认 `True`。
- ****kwargs**：传递给基类 `torchvision.models.resnet.ResNet` 的参数。

```python
class torchvision.models.ResNet50_Weights(value)
```

上面的 model builder 接受以下值作为 `weights` 参数：

- `ResNet50_Weights.DEFAULT` 等价于 `ResNet50_Weights.IMAGENET1K_V2`
- 也可以使用字符串，如 `weights='DEFAULT'` 或 `weights='IMAGENET1K_V1'`

**ResNet50_Weights.IMAGENET1K_V1**



## 参考

- https://pytorch.org/vision/stable/models/resnet.html
