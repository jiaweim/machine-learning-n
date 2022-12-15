# torch.nn

- [torch.nn](#torchnn)
  - [简介](#简介)
  - [容器](#容器)
  - [卷积层](#卷积层)
  - [池化层](#池化层)
  - [填充层](#填充层)
  - [非线性激活（weighted sum, nonlinearity）](#非线性激活weighted-sum-nonlinearity)
  - [非线性激活（其它）](#非线性激活其它)
  - [归一化层](#归一化层)
  - [循环层](#循环层)
  - [Transformer](#transformer)
  - [线性层](#线性层)
  - [Dropout Layers](#dropout-layers)
  - [参考](#参考)

***

## 简介

图的基本构建模块。

## 容器

## 卷积层

## 池化层

## 填充层

## 非线性激活（weighted sum, nonlinearity）

## 非线性激活（其它）

## 归一化层

## 循环层

## Transformer

## 线性层

|Layer|说明|
|---|---|
|

## Dropout Layers

|Layer|说明|
|---|---|
nn.Dropout

During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.

nn.Dropout1d

Randomly zero out entire channels (a channel is a 1D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 1D tensor \text{input}[i, j]input[i,j]).

nn.Dropout2d

Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j]).

nn.Dropout3d

Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj-th channel of the ii-th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j]).

nn.AlphaDropout

Applies Alpha Dropout over the input.

nn.FeatureAlphaDropout

Randomly masks out entire channels (a channel is a feature map, e.g.

## 参考

- https://pytorch.org/docs/stable/nn.html
