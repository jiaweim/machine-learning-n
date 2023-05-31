# PyTorch 安装

## 使用 Wheel 文件安装

文件位置：https://download.pytorch.org/whl

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