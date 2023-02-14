# 张量视图

- [张量视图](#张量视图)
  - [简介](#简介)
  - [View 操作总结](#view-操作总结)
  - [view vs. reshape](#view-vs-reshape)
  - [参考](#参考)

Last updated: 2023-02-13, 10:45
****

## 简介

PyTorch 允许一个张量为另一个已有张量的视图（`View`）。view 张量与其 base 张量共享底层数据，避免了显式复制数据，从而能够快速实现 reshape、切片和逐元素操作。

例如，调用 `t.view(...)` 获得张量 `t` 的视图：

```python
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
>>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` 共享底层数据
True
# 修改 view 张量会影响原张量
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

由于 view 张量与其 base 张量共享底层数据，因此编辑 view 中的数据，base 张量也随之改变。

PyTorch 操作通常返回一个新的张量作为输出，如 `add()`。但是视图操作返回输入张量的视图，避免不必要的数据复制。在创建 view 时没有数据移动，**view 张量只是更改解释相同数据的方式**。

PyTorch 的张量还有内存连续和不连续的概念。张量一般是内存连续的，而 view 操作可能产生内存不连续的张量。用户应该注意，内存连续性可能对性能有影响，以 `transpose()` 操作为例：

```python
>>> base = torch.tensor([[0, 1],[2, 3]])
>>> base.is_contiguous()
True
>>> t = base.transpose(0, 1)  # `t` 是 `base` 的 view，因此没有数据移动
# view 张量可能内存不连续
>>> t.is_contiguous()
False
# 调用 `.contiguous()` 可以使张量内存连续，但是会产生数据移动
>>> c = t.contiguous()
```

## View 操作总结

下面是 PyTorch 的所有视图操作：

- 基础切片和索引操作，如 `tensor[0, 2:, 1:7:2]`
- `adjoint()`
- `as_strided()`
- `detach()`
- `diagonal()`
- `expand()`
- `expand_as()`
- `movedim()`
- `narrow()`
- `permute()`
- `select()`
- `squeeze()`
- `transpose()`
- `t()`
- `T`
- `H`
- `mT`
- `mH`
- `real`
- `imag`
- `view_as_real()`
- `unflatten()`
- `unfold()`
- `unsqueeze()`
- `view()`
- `view_as()`
- `unbind()`
- `split()`
- `hsplit()`
- `vsplit()`
- `tensor_split()`
- `split_with_sizes()`
- `swapaxes()`
- `swapdims()`
- `chunk()`
- `indices()` (sparse tensor only)
- `values()` (sparse tensor only)

> **NOTE**：PyTorch 通过索引访问张量内容的行为与 NumPy 相同，即基本索引返回视图，高级索引返回副本。

另外有几个操作需要特别注意：

- `reshape()`, `reshape_as()` 和 `flatten()` 可能返回视图或新的张量，用户的代码不应该依赖于它是否为视图。
- `contiguous()`，如果输入张量已经连续，则直接返回；否则复制数据返回一个新的张量。 

对 PyTorch 内部实现细节，可以参考 ezyang 的博客 [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)。

## view vs. reshape

`view()` 返回原张量的视图，共享底层数据。不过如果所需 view 不是连续的，view 会抛出错误，需要先调用 `tensor.contiguous()` 使张量内存连续。

`reshape()` 会在后台完成以上工作，所以一般建议使用 `reshape()` 而不是 `view()`。

## 参考

- https://pytorch.org/docs/stable/tensor_view.html
- https://zhuanlan.zhihu.com/p/463664495
- 《Deep Learning With Pytorch》, Eli Stevens & Luca Antiga & Thomas Viehmann
