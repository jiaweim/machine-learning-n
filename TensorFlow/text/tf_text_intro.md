# TensorFlow Text 简介

- [TensorFlow Text 简介](#tensorflow-text-简介)
  - [简介](#简介)
  - [安装 TensorFlow Text](#安装-tensorflow-text)
    - [使用 pip 安装](#使用-pip-安装)
    - [从源码构建](#从源码构建)
  - [参考](#参考)

Last updated: 2022-08-08, 09:54
@author Jiawei Mao
****

## 简介

TensorFlow Text 为 TensorFlow 2.0 提供了一系列文本相关的类和操作。该库执行基于文本模型的预处理功能，以及 core TensorFlow 没有提供的用于序列建模的其它功能。

使用 TensorFlow Text 进行文本预处理的优点是它们都在 TensorFlow graph 中完成。因此无需担心训练和推理时的 tokenization 不同，也不用担心预处理有差异。

## 安装 TensorFlow Text

### 使用 pip 安装

安装 TF Text 的版本要和 TF 版本匹配，安装时指定 TF Text 版本：

```powershell
pip install -U tensorflow-text==<version>
```

### 从源码构建

TF Text 的构建环境必须与 TF 相同。因此，如果要手动构建 TF Text，强烈建议同时构建 TF。

如果在 MacOS 上构建，需要安装 coreutils。使用 Homebrew 可能最容易。首先[从源码构建 TF](https://www.tensorflow.org/install/source)。

克隆 TF Text 库：

```powershell
git clone  https://github.com/tensorflow/text.git
```

最后使用构建脚本创建 pip 包：

```powershell
./oss_scripts/run_build.sh
```

## 参考

- https://www.tensorflow.org/text/guide/tf_text_intro
