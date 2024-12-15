# 广播机制

2024-10-22 ⭐
@author Jiawei Mao
***

## 简介

许多 PyTorch 操作支持 NumPy 的广播机制。详情可参考 https://numpy.org/doc/stable/user/basics.broadcasting.html 。

简而言之，对支持广播的 PyTorch 操作，它的 tensor 会自动扩展为相同多大小（无需复制数据）。即对 shape 不同的两个张量也可以执行按元素操作。

## 一般规则

两个 tensor 满足以下条件即可广播：

- 每个 tensor 至少有一个维度
- 在迭代维度大小时，从尾部维度开始，要求满足以下条件之一：
  - 维度相等
  - 其中一个为 1
  - 其中一个没有该维度

例如：

```python
>>> x=torch.empty(5,7,3)
>>> y=torch.empty(5,7,3)
# 相同 shape 显然可广播

>>> x=torch.empty((0,))
>>> y=torch.empty(2,2)
# x 和 y 不可广播，因为 x 一个维度都没有

# 尾部维度可以对齐
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
# x 和 y 可广播
# 尾部维度 1：size 都是 1
# 尾部维度 2: y 的 size 为 1
# 尾部维度 3：x size == y size
# 尾部维度 4：y 没有该维度

# but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
# x 和 y 不可广播，因为尾部维度 3: 2 != 3
```

如果张量 `x` 和 `y` 可广播，按如下规则计算返回 tensor 的 size：

- 如果 `x` 和 `y` 的维数不等，则在维数较小的 tensor 的维数前面加 1，使它们 shape 长度相等
- 然后，对每个维度，最终的维度大小是该维度上 `x` 和 `y` 的**最大值**

```python
# 尾部维度可以对齐
>>> x=torch.empty(5,1,4,1)
>>> y=torch.empty(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

# but not necessary:
>>> x=torch.empty(1)
>>> y=torch.empty(3,1,7)
>>> (x+y).size()
torch.Size([3, 1, 7])

# 尾部维度无法对齐，不能广播
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

## in-place 语义

in-place 操作不允许 in-place tensor 因广播改变 shape。

例如：

```python
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(3,1,1)
>>> (x.add_(y)).size()
torch.Size([5, 3, 4, 1])

# but:
>>> x=torch.empty(1,3,1)
>>> y=torch.empty(3,1,7)
>>> (x.add_(y)).size()
RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.
```

## 向后兼容

PyTorch 的早期版本允许某些 pointwise 函数在具有不同 shape 的张量上执行，只要每个张量的元素数量相等即可。pointwise 操作通过将每个张量视为一维来执行。现在 PyTorch 支持广播，这种将 tensor 视为一维来支持的方式已弃用，对元素数量相同但不可广播的 tensor 会生成警告。

如果两个 tensor 的 shape 不同，但可广播且元素数量相同，则引入广播可能导致向后不兼容。例如：

```python
>>> torch.add(torch.ones(4,1), torch.randn(4))
```

在以前会生成一个大小为 `torch.Size([4,1])` 的 tensor，现在则会生成一个大小为 `torch.Size([4,4])` 的 tensor。为了辅助识别代码中可能因为引入广播导致向后不兼容的情况，可以将 `torch.utils.backcompat.broadcast_warning.enabled` 设置为 `True`，从而在遇到这类情况时生成警告。

例如：

```python
>>> torch.utils.backcompat.broadcast_warning.enabled=True
>>> torch.add(torch.ones(4,1), torch.ones(4))
__main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.
```

## 参考

- https://pytorch.org/docs/stable/notes/broadcasting.html
