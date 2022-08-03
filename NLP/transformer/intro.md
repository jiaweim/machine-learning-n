# Transformer

## 简介

Transformer 相对 LSTM 或 GRU 的优势：

- Transformer 能够利用分布式 GPU 进行并行训练，提升模型训练效率；
- 在分析预测更长的文本时，捕捉间隔较长的语义关联效果更好。

LSTM 和 GRU 的也能捕捉长文本之间的语义，但是捕捉能力弱于 Transformer。

基于 seq2seq 架构的 Transformer 模型可以完成 NLP 领域研究的典型任务，如机器翻译、文本生成等，同时又可以构建预训练语言模型，用于不同任务的迁移学习。

## 架构



## 参考

- https://huggingface.co/course/chapter1/1?fw=pt
