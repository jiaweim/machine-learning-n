# Optimization

- [Optimization](#optimization)
  - [简介](#简介)
  - [AdamW (PyTorch)](#adamw-pytorch)
  - [参考](#参考)


## 简介

`.optimization` 模块提供：

- 权重衰减固定的 optimizer，可用于微调模型
- 继承自 `_LRSchedule` 的多个调度对象
- 累计多个 batch 梯度的梯度累计类

## AdamW (PyTorch)

```python
class transformers.AdamW
```

参数：

- **params** (Iterable[nn.parameter.Parameter])



## 参考

- https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules
