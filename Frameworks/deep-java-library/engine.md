# Engines

## 简介

`Engine` 是 DJL 最重要的类之一。DJL 的大部分核心功能，包括 NDArray, NDManager, Model 都只是接口。它们形成以 `Engine` 类为根的接口树。

这些接口的实现由各引擎通过 Java service-loader 加载。这意味着 DJL 能够充分利用这些引擎中大量的性能优化和硬件支持。而且，DJL 能够自由切换不同引擎。

此外，这些 engines 对生产系统非常有用。在 Python 中用这些引擎训练的模型通常可以通过 DJL 导入到 Java 中运行。这使得集成模型到现有 Java 服务器变得更加容易。因为它们使用训练时使用的模型，性能和准确度都不会有损失。

在 DJL 中训练，engine 的选择不是那么重要。任何完全实现 DJL 规范的引擎都会得到类似结果。鼓励你编写引擎无关的代码，这些可以像切换依赖项一样轻松地切换引擎。一般来说，建议使用下面推荐的引擎，除非你有充分的理由使用其它引擎。

### 支持的引擎

DJL 目前支持的引擎有：

- MXNet - 完全支持
- PyTorch - 完全支持
- TensorFlow - 支持推理和 NDArray 操作
- ONNX Runtime - 支持基本推理
- XGBoost - 支持基本推理
- LightGBM - 支持基本推理

### 设置

要选择引擎，首先要将其添加到 Java classpath。通常这意味着添加 Maven 或 Gradle 依赖项。许多引擎需要添加多个依赖项，相关信息可参看对应引擎的 README。

也可以同时加载多个引擎。在 DJL 启动时，它会从可用的引擎中选择一个作为默认引擎。大多数需要引擎的 API，如 `NDManager.newBaseManager()` 和 `Model.newInstance()` 内部都会使用默认引擎。DJL 根据推荐顺序来选择默认引擎。也可以手动选择引擎，例如 `Engine.getEngine(engineName)`，或者 `NDManager.newBaseManager(engineName)`。

有些调用会利用所有可用的引擎。例如，加载模型会尝试所有可用的引擎，看看是否适合待加载的模型。

也可以手动设置默认引擎。每个引擎都有一个名称，可以在引擎的 javadoc 或 README 中找到。通过设置 `DJL_DEFAULT_ENGINE` 环境变量，或 `ai.djl.default_engine` java 属性设置默认引擎。

## PyTorch

PyTorch 引擎包含 3 个模块：

- PyTorch Engine - PyTorch Engine 的 DJL 实现
- PyTorch Model Zoo - 包含从 PyTorch 导出的模型
- PyTorch native library - 用于构建包含 pytorch native 的工具模块

### PyTorch Engine

该模块包含 PyTorch 的 EngineProvider。

不建议开发者直接使用该模块的类，使用这些类会使代码与 PyTorch 绑定，切换框架变得困难。

依赖项：

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-engine</artifactId>
    <version>0.34.0</version>
    <scope>runtime</scope>
</dependency>
```

从 DJL 0.14.0 起，`pytorch-engine` 可以加载老版本的 pytorch native 库。指定 PyTorch 版本的方法有两种：

1. 显示指定 `pytorch-native-xxx` 包版本，覆盖 BOM 中的版本
2. 设置环境变量：`PYTORCH_VERSION` 以覆盖默认版本

#### 支持的 PyTorch 版本

| PyTorch engine version | PyTorch native library version            |
| :--------------------- | :---------------------------------------- |
| pytorch-engine:0.35.0  | 1.13.1, 2.5.1, **2.7.1**                  |
| pytorch-engine:0.34.0  | 1.13.1, 2.5.1, **2.7.1**                  |
| pytorch-engine:0.33.0  | 1.13.1, 2.1.2, 2.3.1, **2.5.1**           |
| pytorch-engine:0.32.0  | 1.13.1, 2.1.2, 2.3.1, **2.5.1**           |
| pytorch-engine:0.31.0  | 1.13.1, 2.1.2, 2.3.1, 2.4.0, **2.5.1**    |
| pytorch-engine:0.30.0  | 1.13.1, 2.1.2, 2.3.1, **2.4.0**           |
| pytorch-engine:0.29.0  | 1.13.1, 2.1.2, 2.2.2, **2.3.1**           |
| pytorch-engine:0.28.0  | 1.13.1, 2.1.2, **2.2.2**                  |
| pytorch-engine:0.27.0  | 1.13.1, **2.1.1**                         |
| pytorch-engine:0.26.0  | 1.13.1, 2.0.1, **2.1.1**                  |
| pytorch-engine:0.25.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.24.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.23.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.22.1  | 1.11.0, 1.12.1, **1.13.1**, 2.0.0         |
| pytorch-engine:0.21.0  | 1.11.0, 1.12.1, **1.13.1**                |
| pytorch-engine:0.20.0  | 1.11.0, 1.12.1, **1.13.0**                |
| pytorch-engine:0.19.0  | 1.10.0, 1.11.0, **1.12.1**                |
| pytorch-engine:0.18.0  | 1.9.1, 1.10.0, **1.11.0**                 |
| pytorch-engine:0.17.0  | 1.9.1, 1.10.0, 1.11.0                     |
| pytorch-engine:0.16.0  | 1.8.1, 1.9.1, 1.10.0                      |
| pytorch-engine:0.15.0  | pytorch-native-auto: 1.8.1, 1.9.1, 1.10.0 |
| pytorch-engine:0.14.0  | pytorch-native-auto: 1.8.1, 1.9.0, 1.9.1  |
| pytorch-engine:0.13.0  | pytorch-native-auto:1.9.0                 |
| pytorch-engine:0.12.0  | pytorch-native-auto:1.8.1                 |
| pytorch-engine:0.11.0  | pytorch-native-auto:1.8.1                 |
| pytorch-engine:0.10.0  | pytorch-native-auto:1.7.1                 |
| pytorch-engine:0.9.0   | pytorch-native-auto:1.7.0                 |
| pytorch-engine:0.8.0   | pytorch-native-auto:1.6.0                 |
| pytorch-engine:0.7.0   | pytorch-native-auto:1.6.0                 |
| pytorch-engine:0.6.0   | pytorch-native-auto:1.5.0                 |
| pytorch-engine:0.5.0   | pytorch-native-auto:1.4.0                 |
| pytorch-engine:0.4.0   | pytorch-native-auto:1.4.0                 |

#### BOM

强烈建议使用 BOM 管理依赖。

DJL 默认会在第一次运行 DJL 时，将 PyTorch native 库下载到缓存文件夹。它会自动根据平台和 GPU来选择合适的 jar。

> [!CAUTION]
>
> 从 PyTorch 2.7.1 开始不支持 precxx11

#### Amazon Linux 2 support

从 PyTorch 2.7.1 开始不支持 AmazonLinux 2.

如果在很老的系统运行（如 Amazonlinux 2），则必须使用 percxx11 build，或设置系统变量来自动选择 precxx11：

```java
System.setProperty("PYTORCH_PRECXX11","true");
System.setProperty("PYTORCH_VERSION","2.5.1");
```

或者使用系统环境：

```
export PYTORCH_PRECXX11=true
export PYTORCH_VERSION=2.5.1
```

如果无法访问网络，可以根据平台添加离线 native 库。

#### 加载自己的 PyTorch native 库

如果通过 python pipi wheel 安装了 PyTorch，并想使用自己安装的 PyTorch，可以设置 `PYTORCH_LIBRARY_PATH` 环境变量。DJL 会加载其指向的 PyTorch native 库。

```
export PYTORCH_LIBRARY_PATH=/usr/lib/python3.13/site-packages/torch/lib

# Use latest PyTorch version that engine supported if PYTORCH_VERSION not set
export PYTORCH_VERSION=2.XX.X

# Use cpu-precxx11 if PYTORCH_FLAVOR not set
export PYTORCH_FLAVOR=cpu
```

#### Windows

PyTorch 需要 Visual C++。如果在 Windows 上使用 DJL 遇到 UnsatisfiedLinkError，请下载并安装 [Visual C++ 2019 ](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) 并重启。

对 Windows 平台，可以选择 CPU 和 GPU。

**Windows GPU**

- ai.djl.pytorch:pytorch-jni:2.7.1-0.34.0
- ai.djl.pytorch:pytorch-native-cu128:2.7.1:win-x86_64 - CUDA 12.4

这里 djl 要求 CUDA 版本为 12.4.

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cu128</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.7.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.7.1-0.34.0</version>
    <scope>runtime</scope>
</dependency>
```

```sh
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

> [!IMPORTANT]
>
> 通过 pip 安装 PyTorch 不需要独立安装 CUDA，因为 pip 会自动下载对应的 CUDA，但 djl 则需要。且 CUDA 版本与 Torch 版本要兼容。

将 C:\Users\jiawe\.djl.ai\pytorch\2.7.1-cu128-win-x86_64 路径添加系统环境变量 PATH 中。

所以要支持 GPU，需要的操作：

1. 安装显卡驱动
2. 安装与显卡驱动和 PyTorch 版本兼容的 CUDA
3. 将 djl 自动下载的引擎添加到 PATH

**Windows CPU**

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.7.1</version>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.7.1-0.34.0</version>
    <scope>runtime</scope>
</dependency>
```

## Dependency Walker

https://github.com/lhak/Dependencies

用于分析依赖项缺失问题。

## 参考

- https://docs.djl.ai/master/docs/engine.html