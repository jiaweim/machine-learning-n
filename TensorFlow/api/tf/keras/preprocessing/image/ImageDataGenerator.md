# ImageDataGenerator

- [ImageDataGenerator](#imagedatagenerator)
  - [简介](#简介)
  - [参数](#参数)
    - [width_shift_range](#width_shift_range)
    - [fill_mode](#fill_mode)
  - [方法](#方法)
    - [flow_from_directory](#flow_from_directory)
  - [参考](#参考)

2022-01-15, 11:03
***

## 简介

通过**数据增强**生成批量的图像张量数据。

```python
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    interpolation_order=1,
    dtype=None
)
```

## 参数

- `featurewise_center`

Boolean，在数据集上，设置输入均值为 0，feature-wise。

- `samplewise_center`

Boolean，是否将每个样本的均值设置为 0.

- `featurewise_std_normalization`

Boolean，是否将输入除以数据集的标准偏差（std），feature-wise。

- `samplewise_std_normalization`

Boolean，对每个输入，除以其 std。

- `rescale`

缩放因子，默认 `None`。如果设置为 `None` 或 0，无缩放，否则在应用其它转换后，所有值乘以 `rescale` 缩放因子。

- `rotation_range`

Int 类型，随机旋转的最大角度值。例如 `rotation_range=40` 表示随机旋转的最大角度为 40°。

- `height_shift_range`

同 `width_shift_range`。

- `shear_range`

Float。剪切强度（逆时针方向的剪切角，以度表示）

- `zoom_range`

Float 或 [lower, upper]，指定随机缩放的范围。

如果是 Float 类型，此时范围为 [1-zoom_range, 1+zoom_range]。

例如，`zoom_range=0.2` 表示随机缩放的比例最大为 20%.

- `horizontal_flip`

Boolean，随机水平翻转。

- `vertical_flip`

Boolean，随机垂直翻转。

### width_shift_range

水平方向平移。

支持 Float, 1D 数组或 int 类型：

- float: 如果 <1，则为总宽度的比例值；如果 >=1，则为 pixels 值；
- 1-D 数组：数组中的随机元素；
- int: 整数指定像素值区间为 `(-width_shift_range, +width_shift_range)`。

如果 `width_shift_range=2`，则可选值包括整数 [-1, 0, +1]，和使用 1D 数组 `width_shift_range=[-1, 0, +1]` 等价；

如果 `width_shift_range=1.0`，采用浮点数，此时区间为 $[-1.0, +1.0)$。

例如，`width_shift_range=0.2` 表示随机水平平移最大距离为宽度的 20%。

### fill_mode

常量 {"constant", "nearest", "reflect" or "wrap"} 中的一个。默认 'nearest'。输入边界外的点根据该模式进行填充：

- 'constant'， kkkkkkkk|abcd|kkkkkkkk (cval=k)，边界外填充常量；
- 'nearest'，aaaaaaaa|abcd|dddddddd，边界外的点用最近的点填充；
- 'reflect'，abcddcba|abcd|dcbaabcd，以映射翻转的形式填充；
- 'wrap'， abcdabcd|abcd|abcdabcd

`cval`：Float or Int。当 `fill_mode = "constant"` 用来指定填充值。

## 方法

### flow_from_directory

```python
flow_from_directory(
    directory, target_size=(256, 256), color_mode='rgb', classes=None,
    class_mode='categorical', batch_size=32, shuffle=True, seed=None,
    save_to_dir=None, save_prefix='', save_format='png',
    follow_links=False, subset=None, interpolation='nearest'
)
```

从指定路径为参数，生成经过数据增强后的数据，在一个无线循环中无线产生 batch 数据。

- `directory`

string, 目标目录。每个类别对应一个子目录。每个子目录中的 PNG、JPG、BMP、PPM 或 TIF 图像会包含在生成器中。

- `target_size`

`(height, width)` integer tuple，默认为 `(256,256)`。为图像 resize 后的尺寸。

- `color_mode`

可选值包括 "ggrayscale", "rgb", "rgba"。默认为 “rgb”。是否将图像转换为 1，3 或 4 通道。

- `classes`

可选的子目录类别列表，如 `['dogs', 'cats']`。默认 `None`。如果不提供 `classes`，则默认从 `directory` 下的子目录结构推断，每个子目录视为一个类，类的顺序（映射到标签索引）按照 alphanumeric 顺序。类名到类索引的 dict 可以通过 `class_indices` 属性查询。

**`class_mode`**

可选值："categorical", "binary", "sparse", "input" 或 `None`。默认 "categorical"。设置返回的标签数组类型：

- "categorical" 对应 2D one-hot 编码标签；
- "binary" 对应 1D binary 标签；
- "sparse" 对应 1D integer 标签；
- "input" 和输入图像相同的图像，主要用于 autoencoder；
- `None`，不返回标签，生成器只生成图像数据 batch，和 `model.predict()` 一起使用很有用。

**`batch_size`**

批量数据大小，默认 32.

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
