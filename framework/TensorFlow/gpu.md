# GPU 支持

## 要求

使用 TensorFlow GPU 需要各种驱动程序和库。

### 硬件要求

[支持 CUDA 的 GPU 卡](https://developer.nvidia.com/zh-cn/cuda-gpus)列表.

## 软件要求

需要安装以下 NVIDIA 软件：

- NVIDIA GPU 驱动
- CUDA 工具包

### 安装 cuda 出错

卸载 nvidia frameview 后重新安装。

CUDA在大版本下向下兼容。比如你装了CUDA11.5，它支持CUDA11.0-CUDA11.5的环境，但不支持CUDA10.2。所以选版本的时候直接选符合你要求的大版本的最新版就行，比如 10.2, 11.6

### cudnn

下载后解压，然后将 bin 目录加入 PATH.
