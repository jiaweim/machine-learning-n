# torch.nn.Sequential

Last updated: 2022-11-17, 17:08
****

## 简介

```python
torch.nn.Sequential(*args: Module)
torch.nn.Sequential(arg: OrderedDict[str, Module])
```

一个顺序容器。模块按照传入构造函数的顺序添加其中。也可以传入包含模块的 `OrderedDict`。`Sequential` 的 `forward()` 接受输入并传递给包含的第一个模块，后续每个模块接受上一个模块的输出，并输出到下一个模块，最后返回最后一个模块的输出。

`Sequential` 与手动调用模块相比，主要优点在于，整个 `Sequential` 容器可以看做一个模块。

`Sequential` 和 `torch.nn.ModuleList` 的区别在于，`ModuleList` 就是一个存储模块的列表，而 `Sequential` 中的模块是按顺序连接起来的。

例如：

```python
# 用 Sequential 创建一个小的模块。当 `model` 运行时，
# 输入首先传入 `Conv2d(1,20,5)`，其输出又作为第一个 `ReLU`
# 的输入；第一个 `ReLU` 的输出作为 `Conv2d(20,64,5)` 的输入
# 最后，`Conv2d(20,64,5)` 的输出作为第二个 `ReLU` 的输入
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)
```

- 以 `OrderedDict` 作为参数定义模型，功能和上一个模型一样

```python
from collections import OrderedDict

model = nn.Sequential(
    OrderedDict([
        ("conv1", nn.Conv2d(1, 20, 5)),
        ("relu1", nn.ReLU()),
        ("conv2", nn.Conv2d(20, 64, 5)),
        ("relu2", nn.ReLU())
    ])
)
```

## 参考

- https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
