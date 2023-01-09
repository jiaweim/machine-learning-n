# torch.nn.functional

- [torch.nn.functional](#torchnnfunctional)
  - [卷积函数](#卷积函数)
  - [池化函数](#池化函数)
  - [非线性激活函数](#非线性激活函数)
  - [线性函数](#线性函数)
  - [Dropout 函数](#dropout-函数)
  - [Sparse 函数](#sparse-函数)
  - [Loss functions](#loss-functions)
    - [l1\_loss](#l1_loss)
  - [操作](#操作)
    - [one\_hot](#one_hot)
  - [参考](#参考)

***

## 卷积函数

## 池化函数

## 非线性激活函数

## 线性函数

## Dropout 函数

## Sparse 函数

## Loss functions

### l1_loss

```python
torch.nn.functional.l1_loss(input, target, 
    size_average=None, 
    reduce=None, 
    reduction='mean') → Tensor
```



## 操作

### one_hot

```python
torch.nn.functional.one_hot(tensor, num_classes=- 1) → LongTensor
```

对 shape 为 `*` 包含索引的 LongTensor，返回 shape 为 `(*, num_classes)` 的 LongTensor，除了最后一个维度的索引位置为 1，其它地方都是 0.

参数：

- **tensor** (`LongTensor`)：任意 shape 的类别值
- **num_classes** (`int`)：总类别数，`-1` 表示从别类值推断，为输入张量最大类别值+1

例如：

```python
>>> F.one_hot(torch.arange(0, 5) % 3)
tensor([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])
>>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]])
>>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
tensor([[[1, 0, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [1, 0, 0]],
        [[0, 1, 0],
         [0, 0, 1]]])
```


## 参考

- https://pytorch.org/docs/stable/nn.functional.html
