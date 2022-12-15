# RandomHorizontalFlip

Last updated: 2022-12-15, 13:49
****

## 简介

```python
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

以指定概率将图片水平翻转。如果图像为 Tensor，要求 shape 为 [..., H, W]，即最后两个维度为 H 和 W，前面的随意。

参数：

- **p** (`float`) 翻转图片概率，默认 0.5.

## 参考

- https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html
