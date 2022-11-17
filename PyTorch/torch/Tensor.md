# torch.Tensor

- [torch.Tensor](#torchtensor)
  - [方法](#方法)
    - [detach](#detach)
    - [numpy](#numpy)
  - [参考](#参考)

***

## 方法

|方法|说明|
|---|---|
|[Tensor.detach](#detach) |返回一个从当前 graph 中分离出来的新的张量|

### detach

从当前图中分离出来，创建一个新的张量。

返回的张量不需要梯度。

该方法会影响正向模式的 AD 梯度，即结果也不会有正向模式的 AD 梯度。

### numpy

```python
Tensor.numpy(*, force=False) → numpy.ndarray
```

返回张量的 numpy 形式。

如果 `force` 为 `False`（默认），则要求 tensor 位于 CPU、不需要 grad、没有设置耦合位，dtype 和 layout  NumPy 支持才执行转换。返回的 ndarray 和 tensor 共享内存，因此张量和 ndarray 变化同步。

如果 `force` 为 `True`，则等价于 `t.detach().cpu().resolve_conj().resolve_neg().numpy()`。如果 tensor 不在 CPU，或者设置了共轭位或负位，则张量与 ndarray 不共享内存。将 `force` 设置为 `True` 是一种获得张量 ndarray 形式的简单方法。

即设置 `force=True` 返回的 ndarray 可以与张量不共享内存。

## 参考

- https://pytorch.org/docs/stable/tensors.html
