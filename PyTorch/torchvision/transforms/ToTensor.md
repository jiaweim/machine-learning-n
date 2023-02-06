# torchvision.transforms.ToTensor

Last updated: 2023-02-06, 10:20
****

## 简介

```python
class torchvision.transforms.ToTensor
```

将 PIL `Image` 或 `numpy.ndarray` 转换为张量。该转换不支持 torchscript.

将 PIL `Image` 或 `numpy.ndarray` (H x W x C) 转换为 [0.0, 1.0] 范围的`torch.FloatTensor` (C x H x W)，要求：

- PIL `Image` 的模式必须为 (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) 中的一种；
- `numpy.ndarray` 取值范围 [0,255] 或 `dtype = np.uint8`

对其它情况，对返回的张量不进行缩放。

> **NOTE**
> 由于输入图像被缩放到 [0.0, 1.0]，因此在转换 target 图像 mask 值时不应使用该转换。

## 参考

- https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
