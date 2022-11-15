# 预备知识

- [预备知识](#预备知识)
  - [数据操作](#数据操作)
    - [数据操作入门](#数据操作入门)
    - [运算符](#运算符)
    - [广播机制](#广播机制)
    - [索引和切片](#索引和切片)
    - [节省内存](#节省内存)
    - [转换为其它对象](#转换为其它对象)
  - [数据预处理](#数据预处理)
    - [读取数据集](#读取数据集)
    - [处理缺失值](#处理缺失值)
    - [转换为张量](#转换为张量)
  - [线性代数](#线性代数)
    - [标量](#标量)
  - [参考](#参考)

***

## 数据操作

### 数据操作入门

**创建数组**

创建数组需要：

- 形状：例如 3x4 矩阵
- 每个元素的数据类型：例如 32 位浮点数
- 每个元素的值，例如全是 0，或者随机数

**访问元素**

![](images/2022-11-14-13-34-00.png)

**PyTorch 实现**

- 使用 `arange` 创建行向量 `x`，包含以 0 开始的 12 个整数

```python
x = torch.arange(12)
x
```

```txt
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

- 通过张量的 `shape` 属性访问张量的形状（沿每个轴的长度）

```python
x.shape
```

```txt
torch.Size([12])
```

- 使用 `numel()` 方法获得张量中元素的总数

```python
x.numel()
```

```txt
12
```

- `reshape` 改变张量形状，但不改变元素数量和元素值

把 x 从形状 (12,) 的行向量转换为 (3,4) 的矩阵。

```python
X = x.reshape(3, 4)
X
```

```txt
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

- 使用全 0、全 1、其它常量或者特定分布中随机采样的数字来初始化矩阵

创建形状为 (2,3,4) 的张量，所有元素设置为 0：

```python
torch.zeros((2, 3, 4))
```

```txt
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
```

创建形状为 (2,3,4) 的张量，所有元素设置为 1：

```python
torch.ones((2, 3, 4))
```

```txt
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
```

- 创建形状为 (3,4) 的张量，元素从均值为0、标准差为 1 的标准高斯分布中随机采样

```python
torch.randn(3, 4)
```

```txt
tensor([[ 0.2602, -1.1463,  0.2369, -0.5475],
        [ 0.7723, -0.5699,  0.4023, -0.7272],
        [ 0.1569,  0.4193,  1.3748,  1.0034]])
```

- 使用包含数值的 Python 列表（或嵌套列表），为张量的每个元素赋值

```python
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```txt
tensor([[2, 1, 4, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1]])
```

### 运算符

常见的标准算数运算符（+、-、*、/ 和 **）都可以升级为按元素运算。同一形状的任意两个张量都可以调用按元素操作。

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# 分别按元素加、减、乘、除和幂
x + y, x - y, x * y, x / y, x**y
```

```txt
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
```

- 按元素计算指数

```python
torch.exp(x)
```

```txt
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

- 连接张量

按行（轴-0）和按列（轴-1）连结两个矩阵

```python
# 生成 2 个 3x4 矩阵
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 以两种方式连结起来
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```txt
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
```

- 通过逻辑运算符构建二维张量

```python
X == Y
```

```txt
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

- 对张量的所有元素求和，产生一个只有一个元素的张量

```python
X.sum()
```

```txt
tensor(66.)
```

### 广播机制

即使形状不同，也可以通过广播机制执行按元素操作。其机制为：通过适当复制来扩展一个或两个数组，使两个张量具有相同形状，然后对生成的数组执行按元素操作。

通过沿着数组中长度为 1 的轴进行广播。例如：

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```txt
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
```

由于 a 和 b 分别是 $3\times 1$ 和 $1\times 2$ 矩阵，如果相加，它们的形状不匹配。如果将两个矩阵广播为一个更大的 $3\times 2$矩阵，就可以相加：

```python
a + b
```

```txt
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

### 索引和切片

与 Python 数组一样，第一个元素的索引是 0，最后一个元素索引是 -1。

- 例如，用 -1 选择最后一个元素，用 [1:3] 选择第二个和第三个元素

```python
X[-1], X[1:3]
```

```txt
(tensor([ 8.,  9., 10., 11.]),
 tensor([[ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]]))
```

- 通过指定索引将元素写入矩阵

```python
X[1, 2] = 9
X
```

```txt
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
```

- 为多个元素赋值

```python
X[0:2, :] = 12
X
```

```txt
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```

### 节省内存

**运行一些操作可能会导致为新结果分配内存**。
例如，如果我们用`Y = X + Y`，我们将取消引用`Y`指向的张量，而是指向新分配的内存处的张量。

在下面的例子中，我们用Python的`id()`函数演示了这一点，它给我们提供了内存中引用对象的确切地址。运行`Y = Y + X`后，我们会发现`id(Y)`指向另一个位置。这是因为Python首先计算`Y + X`，为结果分配新的内存，然后使`Y`指向内存中的这个新位置。

```python
before = id(Y)
Y = Y + X
id(Y) == before
```

```txt
False
```

**原地操作**：使用切片表示法将操作的结果分配给先前分配的数组，例如 `Y[:] = <expression>`。

```python
Z = torch.zeros_like(Y)
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))
```

```txt
id(Z): 2403471355712
id(Z): 2403471355712
```

- 如果在后续计算中没有重复使用`X`，也可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。

```python
before = id(X)
X += Y
id(X) == before
```

```txt
True
```

### 转换为其它对象

- 转换为 ndarray

torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```txt
(numpy.ndarray, torch.Tensor)
```

- 将大小为1的张量转换为Python标量，可以调用`item`函数或Python的内置函数。

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```txt
(tensor([3.5000]), 3.5, 3.5, 3)
```

## 数据预处理

介绍使用 pandas 预处理原始数据，并将原始数据转换为张量格式的步骤。

### 读取数据集

创建一个人工数据集，并存储在CSV（逗号分隔值）文件：

```python
import os

os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join("..", "data", "house_tiny.csv")
with open(data_file, "w") as f:
    f.write("NumRooms,Alley,Price\n")  # 列名
    f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")
```

- 从创建的CSV文件中加载原始数据集

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

```txt
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
```

### 处理缺失值

处理缺失的数据，典型的方法包括**插值法**和**删除法**， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

通过位置索引`iloc`，将 `data` 分成`inputs`和`outputs`，其中前者为`data`的前两列，而后者为`data`的最后一列。对于`inputs`中缺少的数值，用同一列的均值替换“NaN”项。

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

```txt
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
```

- 对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

```txt
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

### 转换为张量

现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```txt
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```

## 线性代数

### 标量

标量由只有一个元素的张量表示。下面实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```txt
(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```



## 参考

- https://zh.d2l.ai/chapter_preliminaries/ndarray.html
