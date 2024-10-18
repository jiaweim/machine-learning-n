# PyTorch 安装

## 使用 Wheel 文件安装

文件位置：https://download.pytorch.org/whl

### Conda 安装

CUDA 一般最好选择最新版。

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## GPU

**安装 CUDA**

1. 根据 https://pytorch.org/get-started/locally/ 选择 cuda 版本
2. 下载 cuda: https://developer.nvidia.com/cuda-gpus
3. 安装 cuda
4. 验证 cuda 是否安装好

```powershell
$nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:36:24_Pacific_Standard_Time_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0
```

**安装 cuDNN**

1. 下载：https://developer.nvidia.com/cudnn
2. 解压
3. 将 cudnn/bin 目录添加到环境变量

## 验证安装

运行如下代码：

```python
import torch

x = torch.rand(5, 3)
print(x)
```

输出：

```
tensor([[0.8086, 0.1943, 0.4405],
        [0.1033, 0.7635, 0.2127],
        [0.0595, 0.1819, 0.3198],
        [0.7168, 0.9443, 0.4810],
        [0.1276, 0.5531, 0.7920]])
```

检查 GPU 是否可用：

```python
import torch
torch.cuda.is_available()
```

