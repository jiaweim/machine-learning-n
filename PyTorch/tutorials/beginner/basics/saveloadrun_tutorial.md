# 保存和加载模型

- [保存和加载模型](#保存和加载模型)
  - [简介](#简介)
  - [保存和加载模型权重](#保存和加载模型权重)
  - [保存和加载完整模型](#保存和加载完整模型)
  - [参考](#参考)

Last updated: 2022-11-08, 16:22
****

## 简介

下面介绍如何保存、加载模型状态。

```python
import torch
import torchvision.models as models
```

## 保存和加载模型权重

PyTorch 将模型学习到的参数保存在 `state_dict` 内部状态字典中。可以通过 `torch.save` 保存这些参数：

```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

```txt
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to C:\Users\happy/.cache\torch\hub\checkpoints\vgg16-397923af.pth
  0%|          | 0.00/528M [00:00<?, ?B/s]
```

要加载模型权重，首先要创建相同模型的实例，然后使用 `load_state_dict()` 加载参数。

```python
model = models.vgg16()  # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

```txt
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

> **NOTE**
> 在推理了之前一定要调用 `model.eval()` 方法，以将 dropout 和 batch normalization 层设置为评估模型。不这样做会导致推理结果不一致。

## 保存和加载完整模型

加载模型权重时，需要首先实例化模型类，因为模型定义了网络结构。将 model 而不是 model.state_dict() 传入保存函数，可以将模型一起保存。

```python
torch.save(model, 'model.pth')
```

按如下方式加载模型：

```python
model = torch.load('model.pth')
```

> **NOTE**
> 该方法使用 Python pickle 模块序列化模型，因此它依赖于在加载模型时实际定义的类。

## 参考

- https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
