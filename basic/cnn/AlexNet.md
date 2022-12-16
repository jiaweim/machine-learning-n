# AlexNet

## 简介

AlexNet 网络诞生于 2012 年，是 Hinton 的代表作之一，当年 ImageNet 竞赛的冠军，Top5 错误率为 16.4% 。

AlexNet

- 使用 relu 激活函数，提升了训练速度
- 使用 Dropout 缓解了过拟合

LRN 操作今年使用很少，它的功能与批标准化相似，所以这里选择主流的 BN 操作实现特征标准化。

训练了一个大型的、深度卷积神经网络，用于在 ImageNet LSVRC-2010 竞赛中对 120 万 1000 个不同类别的高分辨图片进行分类。

## TensorFlow 实现



## 参考

- Krizhevsky, A.; Sutskever, I.; Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. Commun. ACM 2017, 60 (6), 84–90. https://doi.org/10.1145/3065386.
