# 术语

- [术语](#术语)
  - [binary classification](#binary-classification)
  - [classification model](#classification-model)
  - [logits](#logits)
  - [log-odds](#log-odds)
  - [multi-class classification](#multi-class-classification)
  - [neutral network](#neutral-network)
  - [recurrent neutral network](#recurrent-neutral-network)
  - [sigmoid function](#sigmoid-function)
  - [softmax](#softmax)

## binary classification

只有两个互斥类别的分类任务。

## classification model

一种区分两个或多个离散类别的机器学习模型称为分类模型。例如，自然语言处理分类模型可以确定输入语句是法语、西班牙语还是意大利语。

## logits

分类模型生成的原始（non-normalized）预测向量，通常使用标准化（normalization）函数进行转换。对多分类模型，通常用 [softmax](#softmax) 函数进行转换。softmax 函数生成一个归一化的概率向量，每个可能的类别有一个概率值。

## log-odds

事件发生概率的对数。

如果事件为二项概率，则 odds 表示成功概率（p）和失败概率（1-p）的比值。例如，假设一个给定事件成功的概率为 90%，失败的概率为 10%。此时 odds 的计算如下：

$$odds=\frac{p}{1-p}=\frac{0.9}{0.1}=9$$

log-odds 就是 odds 的对数，一般采用自然对数。例如：

$$log-odds=ln(9)=2.2$$

log-odds 是 sigmoid 函数的逆函数。

## multi-class classification

包含两个以上类别的分类问题。例如，大约有 128 种枫树，所以对枫树进行分类的模型就是多分类模型。相反，只区分两个类别的模型称为二分类模型（binary classification model）。

## neutral network

受人大脑启发，神经网络由多层（至少一个隐藏层）神经元组成，

## recurrent neutral network

循环神经网络，一种多次运行的神经网络，每次运行的部分都馈入下一次运行。具体来说，上一次隐藏层的运行为这一次相同隐藏层提供了部分部分。循环神经网络对评估序列特别有用，

## sigmoid function

sigmoid 函数将逻辑回归或多项回归的输出（log odds）映射为概率，返回 0 到 1 之间的值。sigmoid 函数的公式如下：

$$y=\frac{1}{1+e^{-\sigma}}$$

在逻辑回归中，σ 为：

$$\sigma=b+w_1x_1+w_2x_2+...+w_nx_x$$

换句话说，sigmoid 函数将 σ 转换为 0 到 1 之间的概率。

在某些神经网络中，sigmoid 函数作为激活函数使用。

## softmax
