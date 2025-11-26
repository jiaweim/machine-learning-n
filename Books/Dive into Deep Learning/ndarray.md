# 数据操作

2025-11-26⭐
@author Jiawei Mao
***
## 简介

djl 使用 n 维数组 ndarray 存储和转换数据，用 `NDArray` 类表示。

## 创建 NDArray

一般称：

- 一维数组为向量
- 二维数量为矩阵
- 高维数组为张量

下面统称为 `NDArray`。

**arange**

创建一个长度为 12 的行向量：其类型为 `int32`

```java
NDManager manager = NDManager.newBaseManager();
NDArray x = manager.arange(12);
System.out.println(x);
```

```
ND: (12) gpu(0) int32
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
```

这里使用 `NDManager` 创建 ndarray。`NDManager` 实现了接口 `AutoClosable`，并管理由它创建的 ndarray 的生命周期。这是为了管理 Java GC 无法控制的 native 内存消耗。通常用 try blocks 包裹 `NDManager`，这样所有 ndarray 都会及时关闭。

```java
try (NDManager manager = NDManager.newBaseManager()) {
    NDArray x = manager.arange(12);
}
```

**shape**

查看 ndarray 的 shape 信息：

```java
x.getShape()
```

```
(12)
```

**size**

获取 ndarray 中的元素总数，即 shape 所有元素的乘积。

```java
x.size()
```

```
12
```

**reshape**

使用 `reshape` 不改变元素数量和值，只改变 shape。将上面 shape 为 `(12,)` 的行向量变为 shape 为 `(3,4)` 的矩阵。

```java
x = x.reshape(3, 4);
System.out.println(x);
```

```
ND: (3, 4) gpu(0) int32
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11],
]
```

在 reshape 时不需要手动指定每个维度。如果目标 shape 是一个矩阵，那么在知道 width 后，根据元素量就可以计算出 height。ndarray 支持该功能，将需要 ndarray 自动推断的维度设置为 -1 即可。因此，`reshape(3,4)` 等价于 `reshape(-1,4)` 或 `reshape(3,-1)`。

**create**

使用 `manager.create(new Shape(3,4))` 创建未初始化的 ndarray。默认类型为 `float32`。

```java
x = manager.create(new Shape(3, 4));
System.out.println(x);
```

```
ND: (3, 4) gpu(0) float32
[[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.],
]
```

通常，我们希望矩阵初始化为 0、1、其它常量，或从特定分布随机抽样。

**zeros**

创建一个所有元素为 0 的张量。

```java
manager.zeros(new Shape(2, 3, 4))
```

```
ND: (2, 3, 4) gpu(0) float32
[[[0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
 ],
 [[0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
 ],
]
```

**ones**

创建一个所有元素为 1 的张量。

```java
manager.ones(new Shape(2, 3, 4))
```

```
ND: (2, 3, 4) gpu(0) float32
[[[1., 1., 1., 1.],
  [1., 1., 1., 1.],
  [1., 1., 1., 1.],
 ],
 [[1., 1., 1., 1.],
  [1., 1., 1., 1.],
  [1., 1., 1., 1.],
 ],
]
```

**randomNormal**

在构造神经网络参数时，通常会采用随机初始化。下面创建一个 shape 为 `(3,4)` 的 ndarray，其元素值从均值为 0、方差为 1 的标准高斯分布随机采样得到：

```java
manager.randomNormal(0f, 1f, new Shape(3, 4), DataType.FLOAT32)
```

```
ND: (3, 4) gpu(0) float32
[[ 0.9423,  0.1963, -0.292 ,  1.7739],
 [ 0.0696,  1.7184,  0.1135,  1.574 ],
 [ 1.6411,  1.1624, -1.9151, -0.729 ],
]
```

也可以只传入 shape，默认均值为 0、方差为 1，类型为 float32.

```java
manager.randomNormal(new Shape(3, 4))
```

```
ND: (3, 4) gpu(0) float32
[[ 0.1171,  0.7113,  0.288 ,  0.4813],
 [-0.7449,  0.1307, -1.3045,  0.5962],
 [-0.588 , -0.4861, -0.8194, -1.3137],
]
```

**create**

也可以提供每个元素值和 shape 来创建张量。

```java
manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4))
```

```
ND: (3, 4) gpu(0) float32
[[2., 1., 4., 3.],
 [1., 2., 3., 4.],
 [4., 3., 2., 1.],
]
```

## 操作

ndarray 最简单也最有用的运算是 element-wise 运算，对数组的元素执行标准的标量操作。对输入为两个数组的函数，element-wise 运算对来个那个数组每对运算应用标准的二元运算符。

在数学中：

- 一元标量运算（unary scalar operator）表示为 $f:\Reals\rightarrow\Reals$，表示该函数是从一个实数映射到另一个实数
- 二元标量运算（binary scalar operator）表示为 $f:\Reals,\Reals\rightarrow\Reals$

给定两个 shape 相同的向量 $\vec{u}$ 和 $\vec{v}$，以及一个二元运算符 $f$，可以得到向量 $\vec{c}=F(\vec{u},\vec{v})$，其中 $c_i\leftarrow f(u_i,v_i)$，$c_i$, $u_i$, $v_i$ 为向量 $\vec{c}$, $\vec{u}$ 和 $\vec{v}$ 的第 i 个元素。

在 DJL 中，常见的算数运算（+, -, *, /, `**`）都被实现为任意相同 shape 张量的逐元素运算。对任意两个 shape 相同的张量都可以调用 elementwise 运算。

**示例**：创建两个长度为 5 的一维向量，然后执行算术运算

```java
NDArray x = manager.create(new float[]{1f, 2f, 4f, 8f});
NDArray y = manager.create(new float[]{2f, 2f, 2f, 2f});
System.out.println(x.add(y));
```

```
ND: (4) gpu(0) float32
[ 3.,  4.,  6., 10.]
```

```java
System.out.println(x.sub(y)); // 减法
System.out.println(x.mul(y)); // 乘法
System.out.println(x.div(y)); // 除法
System.out.println(x.pow(y)); // 指数
```

```
ND: (4) gpu(0) float32
[-1.,  0.,  2.,  6.]

ND: (4) gpu(0) float32
[ 2.,  4.,  8., 16.]

ND: (4) gpu(0) float32
[0.5, 1. , 2. , 4. ]

ND: (4) gpu(0) float32
[ 1.,  4., 16., 64.]
```

还有很多 elementwise 操作，包括指数等一元运算符。

```java
System.out.println(x.exp());
```

```
ND: (4) gpu(0) float32
[ 2.71828175e+00,  7.38905621e+00,  5.45981483e+01,  2.98095801e+03]
```

除了 elementwise 操作，还可以执行线性代数运算，如向量点积、矩阵乘法等，后面单独讨论。

**concatenate**

可以将多个 ndarray 拼接在一起，堆叠成一个更大的 ndarray。

- 默认沿 axis=0 拼接

```java
NDArray x = manager.arange(12f).reshape(3, 4);
NDArray y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1},
        new Shape(3, 4));
```

```
ND: (6, 4) gpu(0) float32
[[ 0.,  1.,  2.,  3.],
 [ 4.,  5.,  6.,  7.],
 [ 8.,  9., 10., 11.],
 [ 2.,  1.,  4.,  3.],
 [ 1.,  2.,  3.,  4.],
 [ 4.,  3.,  2.,  1.],
]
```

`(3,4)` + `(3,4)` = `(6,4)`

- 指定 axis 拼接

```java
System.out.println(x.concat(y, 1));
```

```
ND: (3, 8) gpu(0) float32
[[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
 [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
 [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.],
]
```

`(3,4)` + `(3,4)` = `(3,8)`。

**boolean 运算**

例如，当 elementwise 相等运算：

```java
System.out.println(x.eq(y));
```

```
ND: (3, 4) gpu(0) boolean
[[false,  true, false,  true],
 [false, false, false, false],
 [false, false, false, false],
]
```

**sum**

将 ndarray 中元素元素相加，得到一个只包含一个元素的 ndarray：

```java
System.out.println(x.sum());
```

```
ND: () gpu(0) float32
66.
```

## 广播机制

前面已经展示如何对两个 shape 相同的 ndarray 进行 elementwise 操作。在特定条件下，即使 shape 不同，也可以通过广播机制进行 elementwise 操作。原理：通过复制一个或两个数组的元素，使得它们 shape 相同，然后执行 elementwise 操作。

例如：

```java
NDArray a = manager.arange(3f).reshape(3, 1);
NDArray b = manager.arange(2f).reshape(1, 2);
System.out.println(a);
System.out.println(b);
```

```
ND: (3, 1) gpu(0) float32
[[0.],
 [1.],
 [2.],
]

ND: (1, 2) gpu(0) float32
[[0., 1.],
]

```

`a` 的 shape 为 `(3,1)`，`b` 的 shape 为 `(1,2)`，将两个 ndarray 广播称一个更大的 `(3,2)` 矩阵，`a` 通过复制 columns 实现，`b` 通过复制 rows 实现。然后就可以相加：

```java
System.out.println(a.add(b));
```

```
ND: (3, 2) gpu(0) float32
[[0., 1.],
 [1., 2.],
 [2., 3.],
]
```

这里自动对 `a` 和 `b` 进行了广播。

## 索引和切片

DJL 的索引和切片语法与 Numpy 相同。与其他 Python 数组一样：

- ndarray 支持通过索引访问元素
- 第一个元素的索引为 0
- 切片包含从开始索引到结束索引之前的所有元素
- 通过负数索引可以从末尾开始选择

**示例**：`x` 的 shape 为 `(3,4)`，切片默认从第一个维度开始，从 ":-1" 切片得到 `(2,4)`

```java
NDArray x = manager.arange(12f).reshape(3, 4);
System.out.println(x.get(":-1"));
```

```
ND: (2, 4) gpu(0) float32
[[0., 1., 2., 3.],
 [4., 5., 6., 7.],
]
```

```java
x.get("1:3")
```

```
ND: (2, 4) gpu(0) float32
[[ 4.,  5.,  6.,  7.],
 [ 8.,  9., 10., 11.],
]
```

**写入**

除了读取数据，还可以通过索引写入元素。

```java
NDArray x = manager.arange(12f).reshape(3, 4);
x.set(new NDIndex("1,2"), 9);
System.out.println(x);
```

```
ND: (3, 4) gpu(0) float32
[[ 0.,  1.,  2.,  3.],
 [ 4.,  5.,  9.,  7.],
 [ 8.,  9., 10., 11.],
]
```

如果想给多个元素赋予相同的值，只需索引这些元素，赋予相同的值。例如，`[0:2, :]` 访问第一行和第二行，其中 `:` 取 axis-1 (column) 的所有元素。虽然这里讨论的是矩阵索引，但也适用于向量和张量。

```java
NDArray x = manager.arange(12f).reshape(3, 4);
x.set(new NDIndex("0:2, :"), 2);
```

```
ND: (3, 4) gpu(0) float32
[[ 2.,  2.,  2.,  2.],
 [ 2.,  2.,  2.,  2.],
 [ 8.,  9., 10., 11.],
]
```

## 内存开销

运行操作可能需要主机分配内存保存结果。例如，`y=x.add(y)`，会取消 `y` 对原 ndarray 的 引用，而将 `y` 指向新分配的内存。

这不是我们所期望的，因为：1. 我们不想分配不必要的内存。机器学习中可能有上百兆字节的参数，并且每秒多次更新这些参数。一般我们希望原地更新。2. 多个变量可能指向相同参数，如果没有原地更新参数，其它引用仍指向旧内存地址，使得部分代码引用过时参数。

在 DJL 中执行就地操作很容易。使用对应的原地运算符即可，如 `addi`, `subi`, `muli` 和 `divi`。

```java
NDArray x = manager.arange(12f).reshape(3, 4);
NDArray y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));

var original = manager.zeros(y.getShape());
var actual = original.addi(x);
System.out.println(actual == original);
```

```
true
```

## 总结

- DJL 的 ndarray 是对 NumPy ndarray 的扩展，性能更好，更适合深度学习
- DJL 的 ndarray 提供许多功能，包括基本数学运算、广播、索引、切片、in-place 操作等

## 参考

- https://d2l.djl.ai/chapter_preliminaries/ndarray.html