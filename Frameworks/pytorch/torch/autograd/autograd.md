# torch.autograd

## 简介

`torch.autograd` 提供任意标量值函数的自动微分的类和函数。

它只需要对现有代码进行少量更改：在声明 `Tensor` 时使用 `requires_grad=True` 关键字参数。目前只支持 float 类型 `Tensor`: half, float, double, bfloat16 和 complex 类型 `Tensor`: cfloat, cdouble。

- `backward`

计算指定张量相对 leaf 的梯度之和。

- `grad`

计算并返回输出相对于出入的梯度之和。

### backward

```python
torch.autograd.backward(tensors, 
    grad_tensors=None, 
    retain_graph=None, 
    create_graph=False, 
    grad_variables=None, 
    inputs=None)
```

计算给定张量相对 graph-leaf 的梯度之和。

graph 使用链式法则进行微分。如果 `tensors` 包含非标量张量且需要梯度，则计算 Jacobian-vector product，此时还需要指定 `grad_tensors`。它是一个长度匹配的序列，包含 Jacobian-vector product 中的 vector，通常是微分函数对应张量的梯度（对不需要梯度张量的张量，可以使用 `None`）。

**参数：**

- **tensors** (*Sequence**[*[*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*] or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 需要求导的张量
- **grad_tensors** (*Sequence**[*[*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or* *None**] or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – 雅克比向量积中的向量，通常为张量每个元素对应的梯度。对标量张量或不需要梯度的张量指定为 `None`。如果所有梯度张量都接受 `None`，则可以不指定该参数
- **retain_graph** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否保留计算图。大多情况不需要将其设置为 `True`，默认为 `create_graph` 值.
- **create_graph** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – 是否创建计算图，用于计算高阶导数。默认 `False`.
- **inputs** (*Sequence**[*[*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*] or* [*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or* *Sequence**[*[*GradientEdge*](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.GradientEdge)*]**,* *optional*) – Inputs w.r.t. which the gradient be will accumulated into `.grad`. All other Tensors will be ignored. If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the `tensors`.


## forward 模式自动微分

> [!WARNING]
>
> 此 API 处于测试阶段。尽管函数签名基本不会变化，但正在改进其运算符覆盖范围。

## autograd graph

autograd 提供了许多检查 graph，以及在反向传播时插入行为的方法。

如果一个张量是 autograd 记录操作的输出（即启用了 grad_mode 并且至少有一个输入需要梯度），则 `torch.Tensor` 的 `grad_fn` 属性保存一个 `torch.autograd.graph.Node`，否则为 `None`。

| 属性 | 说明 |
| ---- | ---- |
|  `graph.Node.name`   |  名称    |
|   `graph.Node.metadata`   |  metadata    |
|   `graph.Node.next_functions`   |      |
| `graph.Node.register_hook` | 注册一个反向 hook |
| `graph.Node.register_prehook` | 注册一个前向 hook |
| `graph.increment_version` | |




## 参考

- https://pytorch.org/docs/stable/autograd.html