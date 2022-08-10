# Transformer

## 简介

概括：

- 基于编码器-解码器架构来处理序列对
- 跟使用注意力的 seq2seq 不同，Transformer 纯基于注意力

Transformer 的主要优势：

- 在处理一组对象时，不对空间或时间关系进行任何假设
- 不需要像 RNN 那样对输入进行串行计算，可以进行并行计算

## 多头注意力

对同一 key, value, query，希望抽取不同的信息。

## 参考

- http://peterbloem.nl/blog/transformers
- https://github.com/jsbaan/transformer-from-scratch
