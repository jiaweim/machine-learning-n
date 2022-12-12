# 张量视图

Last updated: 2022-12-12, 10:58
****

## 简介

PyTorch 允许一个张量为另一个已有张量的视图（`View`）。视图张量与其 base 张量共享底层数据，从而避免了显式复制数据，从而能够快速实现 reshape、切片和逐元素操作。

例如，调用 `t.view(...)` 获得张量 `t` 的视图：

```python
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
>>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` 共享底层数据
True
# 修改视图张量，会同时修改原张量
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

由于视图张量与其 base 张量共享底层数据，因此编辑视图中的数据，base 张量也随之改变。

PyTorch 操作通常返回一个新的张量作为输出，如 `add()`。但是视图操作，返回的输入张量的视图，以避免不必要的数据复制。在创建视图时没有数据移动，视图张量只是更改解释相同数据的方式。连续张量的视图可能生成不连续张量。用户应该注意，连续性可能对性能有影响，以 `transpose()` 为例：

```python
>>> base = torch.tensor([[0, 1],[2, 3]])
>>> base.is_contiguous()
True
>>> t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
# View tensors might be non-contiguous.
>>> t.is_contiguous()
False
# To get a contiguous tensor, call `.contiguous()` to enforce
# copying data when `t` is not contiguous.
>>> c = t.contiguous()
```

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

## 参考

- https://pytorch.org/docs/stable/tensor_view.html
