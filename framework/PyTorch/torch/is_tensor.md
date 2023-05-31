# torch.is_tensor

Last updated: 2022-11-17, 19:49
****

## 简介

```python
torch.is_tensor(obj)
```

如果 `obj` 是 PyTorch tensor，返回 `True`。

这个函数只是执行 `isinstance(obj, Tensor)`。

例如：

```python
>>> x = torch.tensor([1, 2, 3])
>>> torch.is_tensor(x)
True
```

## 参考

- https://pytorch.org/docs/stable/generated/torch.is_tensor.html
