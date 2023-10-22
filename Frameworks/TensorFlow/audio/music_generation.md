# 使用 RNN 生成音乐

## 简介

下面演示使用简单的 RNN 生成音符。使用来自 [MAESTRO 数据集](https://magenta.tensorflow.org/datasets/maestro)的钢琴 MIDI 文件训练模型。给定一个音符序列，模型预测下一个音符。通过反复调用模型，可以生成更长的音符序列。

本教程包含解析和创建 MIDI 文件的完整代码。要了解有关 RNN 工作方式的更多信息，请参考 [Text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation)。

## 配置

## 参考

- https://www.tensorflow.org/tutorials/audio/music_generation
