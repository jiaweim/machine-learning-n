# TorchScript 简介

- [TorchScript 简介](#torchscript-简介)
  - [1. 概述](#1-概述)
  - [2. PyTorch 模型基础](#2-pytorch-模型基础)
  - [3. TorchScript 基础](#3-torchscript-基础)
    - [3.1 Tracing](#31-tracing)
  - [4. 使用 Script 转换 Module](#4-使用-script-转换-module)
    - [4.1 混合 Scripting 和 Tracing](#41-混合-scripting-和-tracing)
  - [5. 保存和加载模型](#5-保存和加载模型)
  - [6. 参考](#6-参考)

Last updated: 2023-01-06, 18:57
****

## 1. 概述

TorchScript 是 PyTorch 模型（`nn.module` 的子类）的中间表示，可以在 C++ 等高性能环境中运行。

下面介绍如下内容：

1. PyTorch 模型创建基础，包括：
   - Module
   - 定义 `forward` 函数
   - 组合 Module

2. 将 PyTorch 模块转换为 TorchScript，TorchScript 是 PyTorch 的高性能部署工具
   - tracing 模块
   - 使用 scripting 编译模块
   - 组合这两种方法
   - 保存和加载 TorchScript 模块

```python
import torch
print(torch.__version__)
```

```txt
1.13.1+cu116
```

## 2. PyTorch 模型基础

`Module` 是 PyTorch 的基本组成单元，包含：

- 构造函数，用于准备相关模块
- 参数和子模块，它们在构造函数中初始化
- `forward` 函数，调用模块时运行的代码

下面定义一个简单的 `Module`：

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))
```

```txt
(tensor([[0.8092, 0.7060, 0.8596, 0.9457],
        [0.2957, 0.7843, 0.7290, 0.8392],
        [0.8291, 0.7217, 0.7831, 0.7325]]), tensor([[0.8092, 0.7060, 0.8596, 0.9457],
        [0.2957, 0.7843, 0.7290, 0.8392],
        [0.8291, 0.7217, 0.7831, 0.7325]]))
```

解释：

- 创建继承 `torch.nn.Module` 的类
- 定义构造函数，上面的构造函数只是调用了 `super` 构造函数
- 定义 `forward` 函数，它接受两个输入，并返回两个输出。

然后实例化模块，定义 `x` 和 `h`（两个 3x4 随机矩阵），调用 `my_cell(x, h)` 指定 `forward` 函数。

再定义一个稍微复杂的模块：

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

```txt
MyCell(
  (linear): Linear(in_features=4, out_features=4, bias=True)
)
(tensor([[0.7797, 0.6947, 0.6556, 0.9265],
        [0.4286, 0.5293, 0.4256, 0.9151],
        [0.8855, 0.3305, 0.6234, 0.8107]], grad_fn=<TanhBackward0>), tensor([[0.7797, 0.6947, 0.6556, 0.9265],
        [0.4286, 0.5293, 0.4256, 0.9151],
        [0.8855, 0.3305, 0.6234, 0.8107]], grad_fn=<TanhBackward0>))
```

相对第一个模型额外添加了 `self.linear` 属性，并在 `forward` 函数中调用。

`torch.nn.Linear` 是来自 PyTorch 标准库的一个 `Module`，和 `MyCell` 一样可以直接调用。所以 `MyCell` 是包含其它 `Module` 的 `Module`。打印 `Module` 可以看到其层次结构。从上面的输出可以看到 `MyCell` 包含 `Linear` 子模块及其参数。

以这种方式组合使用 `Module`，易于编写易读且可重用组件。

定义一个更复杂的模块，其中保留控制流：

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

```txt
MyCell(
  (dg): MyDecisionGate()
  (linear): Linear(in_features=4, out_features=4, bias=True)
)
(tensor([[ 0.1716,  0.8713,  0.5216,  0.9028],
        [ 0.0292,  0.7472, -0.1129,  0.9415],
        [ 0.6663,  0.6277,  0.3400,  0.8248]], grad_fn=<TanhBackward0>), tensor([[ 0.1716,  0.8713,  0.5216,  0.9028],
        [ 0.0292,  0.7472, -0.1129,  0.9415],
        [ 0.6663,  0.6277,  0.3400,  0.8248]], grad_fn=<TanhBackward0>))
```

这里添加了 `MyDecisionGate`，其中包含控制流。

许多深度学习框架采用符号式编程，但 PyTorch 采用的命令式编程，命令式编程使用更容易，但符号式编程运行效率更高，更容易移植。

![](images/dynamic_graph.gif)

## 3. TorchScript 基础

TorchScript 捕获模型定义，下面从 **tracing** 开始说起。

### 3.1 Tracing

下面是一个简单模型：

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)
```

```txt
MyCell(
  original_name=MyCell
  (linear): Linear(original_name=Linear)
)
(tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
         [-0.1392,  0.2829, -0.5304,  0.4139],
         [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>),
 tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
         [-0.1392,  0.2829, -0.5304,  0.4139],
         [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>))
```

注意其中 `torch.jit.trace` 的调用，传入定义的 `Module` 和示例输入，`torch.jit.trace` 以示例数据执行传入的模块 `my_cell`，记录 `my_cell` 运行期间执行的所有操作，并根据这些操作创建 `torch.jit.ScriptModule` 实例。

TorchScript 将捕获的模块转换为中间表示（Intermediate Representation, IR），在深度学习中一般将 IR 称为 graph，即 TorchScript 将捕获的模块转换为 graph。可以使用 `.graph` 属性查看：

```python
traced_cell.graph
```

```txt
graph(%self.1 : __torch__.MyCell,
      %x : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %h : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %linear : __torch__.torch.nn.modules.linear.___torch_mangle_2.Linear = prim::GetAttr[name="linear"](%self.1)
  %20 : Tensor = prim::CallMethod[name="forward"](%linear, %x)
  %11 : int = prim::Constant[value=1]() # C:\Users\happy\AppData\Local\Temp\ipykernel_26112\260609686.py:7:0
  %12 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::add(%20, %h, %11) # C:\Users\happy\AppData\Local\Temp\ipykernel_26112\260609686.py:7:0
  %13 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::tanh(%12) # C:\Users\happy\AppData\Local\Temp\ipykernel_26112\260609686.py:7:0
  %14 : (Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu), Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu)) = prim::TupleConstruct(%13, %13)
  return (%14)
```

graph 这种底层表示对大多数用户来说难以理解，`.code` 属性给出的 Python 语法解释更直观：

```python
print(traced_cell.code)
```

```txt
def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  linear = self.linear
  _0 = torch.tanh(torch.add((linear).forward(x, ), h))
  return (_0, _0)
```

使用 `torch.jit.trace` 捕获 `Module` 的好处有：

1. TorchScript 代码可以在它自己的解释器中调用（可以看作一个定制的 Python 解释器），该解释器不需要全局锁，可以同时处理多个请求。
2. 可以保存整个模型，并在其它环境加载，如在 C++ 中运行。
3. TorchScript 生成的 IR 代码可以优化，从而提高执行效率。
4. TorchScript 支持与多个后端进行交互。

调用 `traced_cell` 生成的结果应该与原 Python 模块一致：

```python
print(my_cell(x, h))
print(traced_cell(x, h))
```

```txt
(tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
        [-0.1392,  0.2829, -0.5304,  0.4139],
        [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>), tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
        [-0.1392,  0.2829, -0.5304,  0.4139],
        [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>))
(tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
        [-0.1392,  0.2829, -0.5304,  0.4139],
        [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>), tensor([[ 0.0373,  0.6731,  0.2021,  0.8040],
        [-0.1392,  0.2829, -0.5304,  0.4139],
        [-0.1892,  0.4779, -0.3964,  0.1755]], grad_fn=<TanhBackward0>))
```

## 4. 使用 Script 转换 Module

下面是包含控制流的模块：

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)
```

```txt
def forward(self,
    argument_1: Tensor) -> Tensor:
  return torch.neg(argument_1)

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = torch.add((dg).forward((linear).forward(x, ), ), h)
  _1 = torch.tanh(_0)
  return (_1, _1)

TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if x.sum() > 0:
```

查看 `.code` 输出，发现找不到 if-else 语句。因为 Tracing 根据示例输入运行代码，记录操作，并构造一个执行相同操作的 `ScriptModule`，在这个过程中，控制流被删除了，这显然不是我们想要的。

对控制流可以使用 script compiler，它直接分析 Python 源码，并将其转换为 TorchScript。使用 script compiler 转换 `MyDecisionGate`：

```python
scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)
```

```txt
def forward(self,
    x: Tensor) -> Tensor:
  if bool(torch.gt(torch.sum(x), 0)):
    _0 = x
  else:
    _0 = torch.neg(x)
  return _0

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = torch.add((dg).forward((linear).forward(x, ), ), h)
  new_h = torch.tanh(_0)
  return (new_h, new_h)
```

可以看到，`torch.jit.script` 完整捕获了程序的行为。

### 4.1 混合 Scripting 和 Tracing

tracing 和 scripting 各有优缺点，tracing 的优点在于捕获的执行流程更简洁，但是不支持控制流这种动态行为，而 scripting 则可以完整捕获模块，但是没有 tracinng 简洁。因此可以混合使用这种方式，在控制流部分使用 scripting，在余下部分使用 tracing。

混合使用 tracing 和 scripting 有两种方式：

- 方法一：scripting 内嵌 tracing

```python
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
```

```txt
def forward(self,
    xs: Tensor) -> Tuple[Tensor, Tensor]:
  h = torch.zeros([3, 4])
  y = torch.zeros([3, 4])
  y0 = y
  h0 = h
  for i in range(torch.size(xs, 0)):
    cell = self.cell
    _0 = (cell).forward(torch.select(xs, 0, i), h0, )
    y1, h1, = _0
    y0, h0 = y1, h1
  return (y0, h0)
```

- 方法二：tracing 内嵌 scripting

```python
class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)
```

```txt
def forward(self,
    xs: Tensor) -> Tensor:
  loop = self.loop
  _0, y, = (loop).forward(xs, )
  return torch.relu(y)
```

## 5. 保存和加载模型

TorchScript 包括代码、参数、属性和 debug 信息，即包含模型的完整信息，因此使用 TorchScript 可以保存和加载完整模型。

保存和加载上面的 `WrapRNN` 模型：

```python
traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)
```

```txt
RecursiveScriptModule(
  original_name=WrapRNN
  (loop): RecursiveScriptModule(
    original_name=MyRNNLoop
    (cell): RecursiveScriptModule(
      original_name=MyCell
      (dg): RecursiveScriptModule(original_name=MyDecisionGate)
      (linear): RecursiveScriptModule(original_name=Linear)
    )
  )
)
def forward(self,
    xs: Tensor) -> Tensor:
  loop = self.loop
  _0, y, = (loop).forward(xs, )
  return torch.relu(y)
```

可以看到，保存的 TorchScript 模块包含完整的模型结构和代码，独立于 Python 代码，在 C++ 中也可以加载运行。

## 6. 参考

- https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
