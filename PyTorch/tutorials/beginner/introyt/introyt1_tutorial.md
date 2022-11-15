# PyTorch 介绍

- [PyTorch 介绍](#pytorch-介绍)
  - [PyTorch Tensor](#pytorch-tensor)

## PyTorch Tensor

首先导入 pytorch：

```python
import torch
```

创建张量的几种方法：

```python
z = torch.zeros(5, 3)
print(z)
print(z.dtype)
```

```txt
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32
```

这里创建了一个 5x3 的全 0 矩阵，并查看它的数据类型为 float32，这是 PyTorch 的默认类型。

如果要创建整型张量，可以覆盖默认值：

```python
i = torch.ones((5, 3), dtype=torch.int16)
print(i)
```

```txt
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
```

可以看到，如果不是默认类型，在打印张量时会显示类型。

随机初始化权重时常见操作，为 PRNG 提供 seed 可以保证结果的可重复性：

```python
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2  # 每个元素乘以 2
print(twos)

threes = ones + twos  # shape 相同，可以相加
print(threes)
print(threes.shape)

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# shape 不同，不能相加，会报错
r3 = r1 + r2
```

```txt
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[3., 3., 3.],
        [3., 3., 3.]])
torch.Size([2, 3])

RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
```

