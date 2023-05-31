# 数据表示

- [数据表示](#数据表示)
  - [图像](#图像)
    - [加载图像文件](#加载图像文件)
    - [修改布局](#修改布局)
    - [数据归一化](#数据归一化)
  - [3D 图像](#3d-图像)
  - [表格数据](#表格数据)
    - [加载 wine 数据](#加载-wine-数据)
    - [表示打分](#表示打分)
    - [独热编码](#独热编码)
  - [DataLoader](#dataloader)

***

## 图像

在消费级相机中，单个像素值通常使用 8-bit 整数表示。在医疗、科学和工业应用中，则有更高的数字精度，如 12-bit 或 16-bit。从而可以对骨密度、温度或深度等物理信息进行更宽范围和更高灵敏的保存。

### 加载图像文件

例如，用 `imageio` 模块记载 PNG 图像：

```python
>>> import imageio.v3 as iio
>>> img_arr = iio.imread('bobby.jpg')
>>> img_arr.shape
(720, 1280, 3)
```

> **NOTE** imageio 使用统一的 API 处理不同图像文件类型，不过在 PyTorch 中，使用 TorchVision 处理图像和视频更好。imageio 适合用来演示。

`img_arr` 是一个三维 NumPy 数组：宽度、高度和通道。

需要注意的，PyTorch 要求图像数据的布局为 $C\times H\times W$：channels, height, width，与通常存储图像的 [height, width, channel] 有所不同。

### 修改布局

可以使用张量的 `permute` 方法对张量进行重排。例如，对输入张量 $H\times W\times C$，可以使用如下方法将其转换为 $C\times H\times W$：

```python
>>> img = torch.from_numpy(img_arr)
>>> out = img.permute(2, 0, 1)
>>> out.shape
torch.Size([3, 720, 1280])
```

不过要注意，该操作不复制张量数据，即 `out` 与 `img` 共享底层 storage。

其它深度学习框架使用的图像布局不同，TensorFlow 将通道放在最后一维，即 $H\times W\times C$。这种策略与底层有利有弊，不过我们只需要准备图像数据，满足深度学习框架的需求。

对小批量数据，PyTorch 的数据维度为 $N\times C\times H\times W$。

预先分配合适 shape 的张量，比用 stack 构建更高效，例如：

```python
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
```

用 `torch.zeros` 创建合适 shape 的张量。将载入 3 张 RGB 图像，高度和宽度均为 256 像素。注意类型，这里假设图像的每种颜色用 8-bit 整数表示，所以用 `uint8`。然后就可以批量加载图像：

```python
import os

data_dir = '../data/image-cats/'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(filenames):
    img_arr = iio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    # 这里只保留了前 3 个通道，因为有些图像还有一个
    # 表示透明度的 alpha 通道，这里忽略
    img_t = img_t[:3]  
    batch[i] = img_t
```

### 数据归一化

神经网络对 [0,1] 或 [-1,1] 范围的数据训练性能最好。

对图像数据，通常需要将张量转换为浮点数，并将像素值归一化。将像素值缩放到 [0, 1]之间，对 8-bit 像素值，可以除以 255：

```python
batch = batch.float()
batch /= 255.0
```

另一种方式是计算输入数据的平均值和标准差，然后对进行缩放，使均值为 0，方差为 1：

```python
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c]) # 取一个 channel，shape 为 [N, H, W]
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std
```

> **NOTE** 这里只对单个批次的图像进行归一化。在实际处理图像时，提前计算所有训练数据的平均值和标准差，然后根据该信息对所有数据归一化更好。

## 3D 图像

在某些图像，如 CT（计算机断层扫描）的医学成像，需要处理从头到脚堆叠的图像序列，每张图像对应横跨人体的一个切片。在 CT 扫描中，强度代表身体不同部位的密度，如肺、脂肪、水、肌肉以及骨骼，在工作站上显示，从暗到亮密度依次增加。

类似灰度图像，CT 只有一个强度通道，所以在本地数据格式中一般忽略通道维度，所以原始数据通常由三个维度。通过堆叠 2D 切片组成三维张量，我们可以构建三维解剖结构的空间数据。如下图所示：

![](2022-12-13-14-37-21.png)

> CT 扫描切片

CT 3D 图像与一般图像本质上没有区别，只是在 `channel` 后多了一个维度：depth，得到一个 5D 张量 $N\times C\times D\times H\times W$。

下面使用 `imageio` 的 `volread` 模块将一个目录下的所有 DICOM (Digital Imaging and Communications in Medicine) 文件组合为一个 NumPy 3D 数组：

```python
>>> import imageio
>>> dir_path = r"..\volumetric-dicom\2-LUNG 3.0  B70f-04083"
>>> vol_arr = imageio.volread(dir_path, format='DICOM')
>>> vol_arr.shape
Reading DICOM (examining files): 99/99 files (100.0%)
  Found 1 correct series.
Reading DICOM (loading data): 99/99  (100.0%)
(99, 512, 512)
```

再使用 `unsqueeze` 添加 `channel` 维度：

```python
>>> vol = torch.from_numpy(vol_arr).float()
>>> vol = torch.unsqueeze(vol, 0)
>>> vol.shape
torch.Size([1, 99, 512, 512])
```

沿着 `batch` 方向迭代多个 volumn，就获得了 5D 数据集。

## 表格数据

表格数据，假设样本互相独立，不像时间序列有先后关系。每个 column 可能包含数字或文本，即不同列的数据类型可能不同。网络上有大量表格数据，例如 https://github.com/awesomedata/awesome-public-datasets 。

以 Wine Quality 数据集为例，该数据集包含 vinho verde（来自葡萄牙北部的葡萄酒）的化学特征和感受评分。其中白葡萄酒的数据可以从此[下载](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv)。该数据包含 12 列，第一行为标题；前面 11 列为化学性质，最后一列为感受评分，从0（非常差）到 10（优秀）。标题内容：

```txt
"fixed acidity"
"volatile acidity"
"citric acid"
"residual sugar"
"chlorides";
"free sulfur dioxide";
"total sulfur dioxide";
"density";
"pH";
"sulphates";
"alcohol";
"quality"
```

一个可能的机器学习任务：从化学特征预测质量打分。比如，我们假设质量打分会随着硫（sulfur）的减少而提高。

![](2022-12-13-16-19-39.png)

> 假设葡萄酒中硫与质量打分之间的关系

### 加载 wine 数据

Python 提供了许多加载 CSV 文件的选项，其中比较流行的有：

- Python 自带的 `csv` 模块
- NumPy
- Pandas

其中 pandas 效率最高，不过为了减少依赖项，且 NumPy 与 PyTorch 的互操作性更好，所以用 NumPy 加载 CSV：

```python
import csv

wine_path = "winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq_numpy
```

```txt
array([[ 7.  ,  0.27,  0.36, ...,  0.45,  8.8 ,  6.  ],
       [ 6.3 ,  0.3 ,  0.34, ...,  0.49,  9.5 ,  6.  ],
       [ 8.1 ,  0.28,  0.4 , ...,  0.44, 10.1 ,  6.  ],
       ...,
       [ 6.5 ,  0.24,  0.19, ...,  0.46,  9.4 ,  6.  ],
       [ 5.5 ,  0.29,  0.3 , ...,  0.38, 12.8 ,  7.  ],
       [ 6.  ,  0.21,  0.38, ...,  0.32, 11.8 ,  6.  ]], dtype=float32)
```

将 NumPy 数组转换为 PyTorch 张量：

```python
>>> wineq = torch.from_numpy(wineq_numpy)
>>> wineq.shape, wineq.dtype
(torch.Size([4898, 12]), torch.float32)
```

### 表示打分

wine 数据集最后一列打分，可以将其视为连续变量，并执行回归分析；或者将其看作标签，并执行分类任务。这两种方法，都需要将打分单独提出来，作为训练的标签。

```python
>>> data = wineq[:, :-1] # 选择除最后一列的所有数据
>>> data, data.shape
(tensor([[ 7.00,  0.27,  ...,  0.45,  8.80],
         [ 6.30,  0.30,  ...,  0.49,  9.50],
         ...,
         [ 5.50,  0.29,  ...,  0.38, 12.80],
         [ 6.00,  0.21,  ...,  0.32, 11.80]]),
 torch.Size([4898, 11]))
```

```python
>>> target = wineq[:, -1] # 选择最后一列
>>> target, target.shape
(tensor([6., 6.,  ..., 7., 6.]), torch.Size([4898]))
```

如果所述，将 `target` 张量转换为 labels 张量有两种选择，一种是直接将其视为整数向量打分：

```python
>>> target = wineq[:, -1].long()
>>> target
tensor([6, 6,  ..., 7, 6])
```

### 独热编码

将打分编码为整数向量可能是合适的，因为 1 分的酒确实比 3 分的差，有排序关系。这种关系也引入距离概念，如 1 和 3 的距离与 2 和 4 的距离相同，如果打分满足这种关系，那么整数向量编码很合适。

如果打分是离散的，没有隐含的顺序或距离，那么独热编码更合适。当打分值只有整数而没有分数，独热编码也适合。

可以用 `scatter_` 实现 one-hot 编码：

```python
>>> target_onehot = torch.zeros(target.shape[0], 10)
>>> target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
tensor([[0., 0.,  ..., 0., 0.],
        [0., 0.,  ..., 0., 0.],
        ...,
        [0., 0.,  ..., 0., 0.],
        [0., 0.,  ..., 0., 0.]])
```

## DataLoader

mini-batch size 一般选择 2 的指数，如 16, 32, 64, 128，32 最常用，可作为默认选项。

