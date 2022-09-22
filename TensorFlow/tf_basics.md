# TensorFlow 基础知识总结

- [TensorFlow 基础知识总结](#tensorflow-基础知识总结)
  - [创建张量](#创建张量)
  - [常用函数](#常用函数)
    - [理解 axis](#理解-axis)
    - [tf.Variable](#tfvariable)
    - [数学运算](#数学运算)
    - [tf.data.Dataset.from_tensor_slices](#tfdatadatasetfrom_tensor_slices)
    - [tf.GradientTape](#tfgradienttape)
    - [enumerate](#enumerate)
    - [tf.one_hot](#tfone_hot)
    - [tf.nn.softmax](#tfnnsoftmax)
    - [assign_sub](#assign_sub)
    - [tf.argmax](#tfargmax)

## 创建张量

**指定维度**

|维度|方法|
|---|---|
|一维|直接写个数|
|二维|[行, 列]|
|多维|[n, m, j, k,...]|

- 创建一个张量

```python
tf.constant(张量内容, dtype=数据类型(可选))
```

- 将 numpy 数组转换为 Tensor

```python
tf.convert_to_tensor(数据名, dtype=数据类型(可选))
```

- 创建全 0 张量

```python
tf.zeros(维度)
```

- 创建全 1 张量

```python
tf.ones(维度)
```

- 创建全为指定值的张量

```python
tf.fill(维度, 指定值)
```

- 生成正态分布的随机数，默认均值为 0，标准差为 1

```python
tf.random.normal(维度, mean=均值, stddev=标准差)
```

- 生成截断式整体分布随机数

```python
tf.random.truncated_normal(维度, mean=均值, stddev=标准差)
```

在 `tf.random.truncated_normal` 中如果随机生成的数据取值在 $(\mu-2\sigma, \mu + 2\sigma)$ 之外，则重新进行生成，保证生成值在均值附近。

- 生成均匀分布随机数 [minval, maxval)

```python
tf.random.uniform(维度, minval=最小值, maxval=最大值)
```

## 常用函数

- 强制数据类型转换

```python
tf.cast(张量名, dtype=数据类型)
```

- 计算张量维度上元素的最小值

```python
tf.reduce_min(张量名)
```

- 计算张量维度上元素的最大值

```python
tf.reduce_max(张量名)
```

### 理解 axis

在一个二维张量或数组中，可以通过调整 axis 等于 0 或 1 来控制执行维度。

- `axis=0` 代表跨行（column），而 `axis=1` 代表跨列（row）
- 如果不指定 axis，则所有元素参与计算

例如：

- 计算张量沿指定维度的平均值

```python
tf.reduce_mean(张量名, axis=操作轴)
```

- 计算张量沿指定维度的和

```python
tf.reduce_sum(张量名, axis=操作轴)
```

### tf.Variable

`tf.Variable()` 将变量**标记为可训练**，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数。

```python
tf.Variable(初始值)

w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
```

### 数学运算

**对应元素的四则运算**

- 两个张量的对应元素相加

```python
tf.add(张量1, 张量2)
```

- 两个张量的对应运算相减

```python
tf.subtract(张量1, 张量2)
```

- 两个张量的对应元素相乘

```python
tf.multiple(张量1, 张量2)
```

- 两个张量的对应元素相除

```python
tf.divide(张量1, 张量2)
```

> [!NOTE] 只有维度相乘的张量才可以做四则运算

**平方，次方与开方**

- 元素平方

```python
tf.square(张量名)
```

- 元素 n 次方

```python
tf.pow(张量名, n 次方数)
```

- 元素开方

```python
tf.sqrt(张量名)
```

**矩阵乘**

```python
tf.matmul(矩阵1, 矩阵2)
```

### tf.data.Dataset.from_tensor_slices

切分传入张量的第一维度，生成输入特征/标签对，构建数据集

```python
data = tf.data.Dataset.from_tensor_slices((输入特征, 标签))
```

> NumPy 和 Tensor 都可用该语句读入数据

```python
import tensorflow as tf

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)
```

下面的元素一一配对：

```txt
(<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
(<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
```

### tf.GradientTape

with 结构记录计算过程，gradient 求出张量的梯度。

```python
with tf.GradientTape( ) as tape:
    若干个计算过程
grad=tape.gradient(函数，对谁求导)
```

例如：

```python
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    y = tf.pow(x, 2)
grad = tape.gradient(y, x)
print(grad)
```

```txt
tf.Tensor(6.0, shape=(), dtype=float32)
```

### enumerate

`enumerate` 是python的内建函数，它可遍历每个元素(如列表、元组或字符串)，组合为：索引元素，常在for循环中使用。

```python
enumerate(列表名)
```

例如：

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
```

```txt
0 one
1 two
2 three
```

### tf.one_hot

独热编码（one-hot encoding）：在分类问题中，常用独热码做标签，标记类别：

- 1表示是
- 0表示非

|0 狗尾草鸢尾|1 杂色鸢尾|2 弗吉尼亚鸢尾|
|---|---|---|
|标签：1|
|独热码：（0. 1. 0.）|

`tf.one_hot()` 函数将待转换数据，转换为one-hot形式的数据输出。

```python
tf.one_hot(待转换数据, depth=几分类)
```

例如：

```python
import tensorflow as tf

classes = 3 # 三种类别
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("result of labels1:", output)
print("\n")
```

```txt
result of labels1: tf.Tensor(
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)
```

### tf.nn.softmax

$$Softmax(y_i)=\frac{e^{y_i}}{\sum_{j=0}^n e^{y_i}}$$

```python
tf.nn.softmax(x)
```

使输出符合概率分布。例如：

```python
import tensorflow as tf

x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])
y = tf.matmul(x1, w1) + b1
print("x1.shape:", x1.shape)
print("w1.shape:", w1.shape)
print("b1.shape:", b1.shape)
print("y.shape:", y.shape)
print("y:", y)

##### 以下代码可将输出结果y转化为概率值 #####
y_dim = tf.squeeze(y)  # 去掉y中纬度1（观察y_dim与 y 效果对比）
y_pro = tf.nn.softmax(y_dim)  # 使y_dim符合概率分布，输出为概率值了
print("y_dim:", y_dim)
print("y_pro:", y_pro)

# 请观察打印出的shape
```

```txt
x1.shape: (1, 4)
w1.shape: (4, 3)
b1.shape: (3,)
y.shape: (1, 3)
y: tf.Tensor([[ 1.0099998   2.008      -0.65999985]], shape=(1, 3), dtype=float32)
y_dim: tf.Tensor([ 1.0099998   2.008      -0.65999985], shape=(3,), dtype=float32)
y_pro: tf.Tensor([0.2563381  0.69540703 0.04825491], shape=(3,), dtype=float32)
```

当 n 分类的 n 个输出 $y_0, y_1,..., y_{n-1}$ 通过softmax( ) 函数，便符合概率分布了。即输出值在 [0, 1] 之间，且加和为 1.

例如：

```python
import tensorflow as tf

y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)

print("After softmax, y_pro is:", y_pro)  # y_pro 符合概率分布

print("The sum of y_pro:", tf.reduce_sum(y_pro))  # 通过softmax后，所有概率加起来和为1
```

```txt
After softmax, y_pro is: tf.Tensor([0.25598174 0.6958304  0.0481878 ], shape=(3,), dtype=float32)
The sum of y_pro: tf.Tensor(0.99999994, shape=(), dtype=float32)
```

### assign_sub

- 用于参数的自更新，赋值操作，更新参数的值并返回。
- 调用 `assign_sub` 前，先用 `tf.Variable` 定义变量 `w` 为可训练（可自更新）。

例如：

```python
import tensorflow as tf

x = tf.Variable(4)
x.assign_sub(1)
print("x:", x)  # 4-1=3
```

```txt
x: <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
```

### tf.argmax

返回张量沿指定维度最大值的索引。

```python
tf.argmax(张量名,axis=操作轴)
```

例如：

```python
import numpy as np
import tensorflow as tf

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test:\n", test)
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引
```

```txt
test:
 [[1 2 3]
 [2 3 4]
 [5 4 3]
 [8 7 2]]
每一列的最大值的索引： tf.Tensor([3 3 1], shape=(3,), dtype=int64)
每一行的最大值的索引： tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
```
