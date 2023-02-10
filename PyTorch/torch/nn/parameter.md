# torch.nn.parameter

## torch.nn.parameter.Parameter

```python
class torch.nn.parameter.Parameter(
    data=None, requires_grad=True)
```

用作 module 参数的 `Tensor`。

`Parameter` 是 `Tensor` 的子类，主要特性：当用作 `Module` 属性，会自动添加到 `Module` 的参数列表，在 `parameters()` 迭代器中。使用 `Tensor` 没有该效果。

**参数：**

- **data** (`Tensor`) – 参数张量。
- **requires_grad** (`bool`, optional) – 是否计算梯度。

## 参考
- 