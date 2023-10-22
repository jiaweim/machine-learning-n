# torch.max

## 简介

```python
torch.max(input) → Tensor
```

返回输入张量 `input` 中所有元素的最大值。例如：

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6763,  0.7445, -2.2369]])
>>> torch.max(a)
tensor(0.7445)
```

```python
torch.max(input, dim, keepdim=False, *, out=None)
```

返回 namedtuple `(values, indices)`，其中 `values` 是 `input` 张量在维度 `dim` 每行的最大值，

## 参考

- https://pytorch.org/docs/stable/generated/torch.max.html
