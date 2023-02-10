# torch.nn.Module

## 简介

所有神经网络模块的基类。

所有模型应该继承该类。

`nn.Module` 可以包含其它 `nn.Module`，即支持嵌套：

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

## 方法

### add_module

```python
add_module(name, module)
```

添加一个子 module。

可以使用指定的名称以属性的方式访问添加的模块。

**参数：**

- **name** (`str`) - 子模块名称。可以通过该名称访问子模块。
- **module** (`Module`) – 待添加的模块

### parameters

```python
parameters(recurse=True)
```

返回模块参数的迭代器。

传统传递给 optimizer。

**参数：**

- **recurse** (`bool`)

`True` 则生成该模块及所有子模块的参数。否则，只生成该模块直接成员的参数。

**返回：**

- Iterator[Parameter]

**示例：**

```python
>>> for param in model.parameters():
>>>     print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

### state_dict

```python
state_dict(*, destination: T_destination, prefix: str = '', keep_vars: bool = False) → T_destination
```

```python
state_dict(*, 
    prefix: str = '', 
    keep_vars: bool = False) → Dict[str, Any]
```

返回包含模块状态的 dict。

只包含可学习参数。

### train

```python
train(mode=True)
```

将 module 设置为训练模式。

该设置对部分模块有影响，如 `Dropout`, `BatchNorm` 等。

**参数：**



## 参考

- https://pytorch.org/docs/stable/generated/torch.nn.Module.html
