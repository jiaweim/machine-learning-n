# torch.randn

***

## 简介

```python
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, 
    requires_grad=False, pin_memory=False) → Tensor
```

返回一个由均值为0、方差为 1 的正态分布中随机数填充的张量。

$$out_i ∼ N(0,1)$$

张量的形状由参数 `size` 定义。

## 参数

- **size** (int...)

整数序列，定义输出张量的形状。可以是多个可变参数，也可以是 list 或 tuple。

- **generator** ([torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), optional)

用于抽样的伪随机数生成器。

- **out** (`Tensor`, optional)

输出张量。

- **dtype** (`torch.dtype`, optional)

返回张量的期望数据类型。如果为 `None`，则使用全局默认类型，参考 `torch.set_default_tensor_type()`。



## 示例

```python
>>> torch.randn(4)
tensor([-0.6837, -0.0592,  1.2451, -0.8639])
>>> torch.randn(2, 3)
tensor([[ 0.6635, -1.0228,  0.0674],
        [ 1.4007,  1.6177, -0.7507]])
```

## 参考

- https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn
