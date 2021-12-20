# GoogLeNet

- [GoogLeNet](#googlenet)
  - [简介](#简介)
  - [结构](#结构)
    - [1x1 Conv](#1x1-conv)
  - [参考](#参考)

2021-12-20, 10:34
***

## 简介

GoogLeNet 架构由来自 Google Research 的 Christian Szegedy 等人开发，赢得了 2014 年 ILSVRC 调整，将 top5 error rate 降到 7% 以下。其出色的表现，很大程度是因为它比之前的 CNN 要深得多。

## 结构

### 1x1 Conv

1x1 Conv 的运算如下所示：

![](images/2021-12-20-12-43-27.png)

其主要作用是降低 channel 数目，减少运算量。

## 参考

- https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html
- https://www.bilibili.com/video/BV1Y7411d7Ys
