# 图像变换和增强

- [图像变换和增强](#图像变换和增强)
  - [简介](#简介)
  - [变换组合](#变换组合)
    - [transforms.Compose](#transformscompose)
  - [支持 PIL Image 和 Tensor](#支持-pil-image-和-tensor)
    - [transforms.RandomHorizontalFlip](#transformsrandomhorizontalflip)
    - [transforms.Resize](#transformsresize)
  - [仅支持 PIL Image](#仅支持-pil-image)
  - [仅支持 Tensor](#仅支持-tensor)
    - [transforms.Normalize](#transformsnormalize)
  - [转换变化](#转换变化)
    - [ToTensor](#totensor)
  - [参考](#参考)

Last updated: 2023-02-08, 14:37
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

## 变换组合

### transforms.Compose

```python
class torchvision.transforms.Compose(transforms)
```

将多个变换组合在一起。该转换不支持 torchscript。

**参数：**

- **transforms** (list of `Transform` objects)

要组合的变换 list。

**示例：**

```python
transforms.Compose([
    transforms.CenterCrop(10),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])
```

> **NOTE**
> 为了使变换支持 torchscript，建议按如下方式使用 `torch.nn.Sequential`

```python
transforms = torch.nn.Sequential(
    transforms.CenterCrop(10),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
scripted_transforms = torch.jit.script(transforms)
```

确保只使用 scriptable 的变换，即使用 `torch.Tensor`，不需要 lambda 函数或 `PIL.Image`。

## 支持 PIL Image 和 Tensor


CenterCrop(size)

Crops the given image at the center.

ColorJitter([brightness, contrast, ...])

Randomly change the brightness, contrast, saturation and hue of an image.

FiveCrop(size)

Crop the given image into four corners and the central crop.

Grayscale([num_output_channels])

Convert image to grayscale.

Pad(padding[, fill, padding_mode])

Pad the given image on all sides with the given "pad" value.

RandomAffine(degrees[, translate, scale, ...])

Random affine transformation of the image keeping center invariant.

RandomApply(transforms[, p])

Apply randomly a list of transformations with a given probability.

RandomCrop(size[, padding, pad_if_needed, ...])

Crop the given image at a random location.

RandomGrayscale([p])

Randomly convert image to grayscale with a probability of p (default 0.1).

### transforms.RandomHorizontalFlip

```python
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

以指定概率将图片水平翻转。如果图像为 `Tensor`，要求 shape 为 [..., H, W]，即最后两个维度为 H 和 W，前面的随意。

**参数：**

- **p** (`float`) 翻转图片概率，默认 0.5.

```python
forward(img)
```

**参数：**

- **img** (PIL Image or `Tensor`) – 待翻转 Image

**返回：**

随机翻转后的图像。PIL Image 或 Tensor。



RandomPerspective([distortion_scale, p, ...])

Performs a random perspective transformation of the given image with a given probability.

RandomResizedCrop(size[, scale, ratio, ...])

Crop a random portion of image and resize it to a given size.

RandomRotation(degrees[, interpolation, ...])

Rotate the image by angle.

RandomVerticalFlip([p])

Vertically flip the given image randomly with a given probability.

### transforms.Resize

```python
class torchvision.transforms.Resize(
    size, 
    interpolation=InterpolationMode.BILINEAR, 
    max_size=None, 
    antialias=None)
```

将输入图像调整到指定大小。如果图像是 Tensor 类型，要求 shape 为 `[…, H, W] `。

> **WARNING**
> 根据类型不同输出图像可能不同：向下采样时，PIL 图像和 Tensor 的差值略有不同，PIL 应用了抗锯齿。这可能导致网络性能的显著差异。

**参数：**

- **size** (`sequence` or `int`)

所需输出 size。如果是类似 (h, w) 的序列，输出大小将与此匹配。如果 `size` 为 `int`，则图像较小的边与此匹配，例如，如果 height > width，则将图像缩放为 (size * height / width, size)。

> **NOTE**
> torchscript 不支持单个 `int` 的 `size`，可以用长度为 1 的序列代替：`[size, ]`

- **interpolation** (`InterpolationMode`)

插值策略，由 enum `torchvision.transforms.InterpolationMode` 定义。默认 `InterpolationMode.BILINEAR`。

如果输入 `Tensor`，只支持 `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` 和 `InterpolationMode.BICUBIC`。

- **max_size** (`int`, optional)

调整大小后图像长边允许的最大值：如果根据 `size` 调整大小后图像的长边大于 `max_size`，则再次调整图像大小使长边为 `max_size`。当 `size` 为 int 时（或 torchscript 模式中长度为 1 的 sequence），才支持 `max_size`。

- **antialias** (`bool`, optional)

对 PIL 图像 `img`，忽略该参数并始终使用抗锯齿。
对 Tensor，`antialias` 默认为 `False`，对 `InterpolationMode.BILINEAR` 和 `InterpolationMode.BICUBIC` 模式可以设置为 `True`，这有助于使 PIL 图像和 Tensor 图像的输出更接近。


TenCrop(size[, vertical_flip])

Crop the given image into four corners and the central crop plus the flipped version of these (horizontal flipping is used by default).

GaussianBlur(kernel_size[, sigma])

Blurs image with randomly chosen Gaussian blur.

RandomInvert([p])

Inverts the colors of the given image randomly with a given probability.

RandomPosterize(bits[, p])

Posterize the image randomly with a given probability by reducing the number of bits for each color channel.

RandomSolarize(threshold[, p])

Solarize the image randomly with a given probability by inverting all pixel values above a threshold.

RandomAdjustSharpness(sharpness_factor[, p])

Adjust the sharpness of the image randomly with a given probability.

RandomAutocontrast([p])

Autocontrast the pixels of the given image randomly with a given probability.

RandomEqualize([p])

Equalize the histogram of the given image randomly with a given probability.

## 仅支持 PIL Image

## 仅支持 Tensor

LinearTransformation(transformation_matrix, ...)

Transform a tensor image with a square transformation matrix and a mean_vector computed offline.

### transforms.Normalize

```python
torchvision.transforms.Normalize(mean, std, inplace=False)
```

根据均值和方差归一化张量图像。该变换不支持 PIL 图像。对 `n` 个通道的均值 `(mean[1],...,mean[n])` 和标准差 `(std[1],..,std[n])`，该变换对每个通道依次进行归一化，即 `output[channel] = (input[channel] - mean[channel]) / std[channel]`。

> **NOTE**：该变换不是原地操作，即不改变输入张量。

**参数：**

- **mean** (`sequence`) – 每个通道的均值的序列
- **std** (`sequence`) – 每个通道的标准差序列
- **inplace** (`bool`,optional) – 是否为原地操作

```python
forward(tensor: Tensor) → Tensor
```

**参数：**

- **tensor** (`Tensor`) – 待归一化的图像张量

**返回：**

- 归一化后的图像张量。
- `Tensor` 类型

**示例：**




RandomErasing([p, scale, ratio, value, inplace])

Randomly selects a rectangle region in an torch Tensor image and erases its pixels.

ConvertImageDtype(dtype)

Convert a tensor image to the given dtype and scale the values accordingly This function does not support PIL Image.

## 转换变化

ToPILImage([mode])

Convert a tensor or an ndarray to PIL Image.

### ToTensor

```python
class torchvision.transforms.ToTensor
```

将 PIL `Image` 或 `numpy.ndarray` 转换为张量。不支持 torchscript.

将 PIL `Image` 或 `numpy.ndarray` (H x W x C) 转换为 [0.0, 1.0] 范围的`torch.FloatTensor` (C x H x W)，要求：

- PIL `Image` 的模式必须为 (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) 中的一种；
- `numpy.ndarray` 取值范围 [0,255] 或 `dtype = np.uint8`

对其它情况，对返回的张量不进行缩放。

> **NOTE**
> 由于输入图像被缩放到 [0.0, 1.0]，因此在转换 target 图像 mask 值时不应使用该转换。


PILToTensor()

Convert a PIL Image to a tensor of the same type.

## 参考

- https://pytorch.org/vision/stable/transforms.html
