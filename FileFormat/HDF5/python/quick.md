# 快速入门

- [快速入门](#快速入门)
  - [安装](#安装)
  - [核心概念](#核心概念)
  - [创建文件](#创建文件)
  - [Group 和分层结构](#group-和分层结构)
  - [属性](#属性)
  - [参考](#参考)

Last updated: 2023-02-02, 09:50
****

## 安装

- Anaconda

```powershell
conda install h5py
```

- pip

```powershell
pip install h5py
```

## 核心概念

HDF5 文件是一个容器，包含两种对象：

- datasets，类似数组
- groups，类似文件夹，可以包含 dataset 和 group

在 h5py 中只需要记住一点：像 dict 使用 **group**，像 NumPy 数组使用 **dataset** 。

假设 `mytestfile.hdf5` 是一个 HDF5 文件，你需要做的第一件事是打开文件：

```python
import h5py
f = h5py.File('mytestfile.hdf5', 'r')
```

返回的文件对象是读写 HDF5 文件的起点。`h5py.File` 类似 Python dict，可以检查其 keys：

```python
>>> list(f.keys())
['mydataset']
```

可以发现，该文件只有一个数据集 `mydataset`。查询 `Dataset` 对象：

```python
>>> dset = f['mydataset']
```

- `dset` 不是数组，而是 HDF5 的 `Dataset` 对象。和 NumPy `Dataset` 也有 shape 和 dtype:

```python
>>> dset.shape
(100,)
>>> dset.dtype
dtype('int32')
```

- `Dataset` 支持切片，下面用切片读取和写入数据

```python
>>> dset[...] = np.arange(100)
>>> dset[0]
0
>>> dset[10]
10
>>> dset[0:100:10]
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
```

## 创建文件

创建上面使用的 `mytestdata.hdf5` 文件。文件模式 `mode` 设置为 `w`，另外还有 `a` (read/write/create) 和 `r+` (read/write) 模式。

```python
import h5py
import numpy as np
f = h5py.File("mytestfile.hdf5", "w")
```

创建数据集：

```python
dset = f.create_dataset("mydataset", (100,), dtype='i')
```

- 文件支持上下文管理器，所以可以使用如下代码

```python
import h5py
import numpy as np

with h5py.File("mytestfile.hdf5", "w") as f:
    dset = f.create_dataset("mydataset", (100,), dtype='i')
```

## Group 和分层结构

HDF 指分层数据格式（Hierarchical Data Format）。HDF5 文件中的每个对象都有一个名称，以 posix 样式的分层存放，以 `/` 为分隔符。例如：

```python
>>> dset.name
'/mydataset'
```

HDF5 中的文件夹称为 group。`File` 本身就是一个 group，称为根 group，名称为 `/`：

```python
>>> f.name
'/'
```

- 创建 subgroup 也是用 `create_group`。对新打开的 hdf5 文件，追加数据要用 "append" 模式

```python
>>> f = h5py.File('mydataset.hdf5', 'a')
>>> grp = f.create_group("subgroup")
```

`Group` 同 `File` 一样，也有 `create_*` 方法：

```python
>>> dset2 = grp.create_dataset("another_dataset", (50,), dtype='f')
>>> dset2.name
'/subgroup/another_dataset'
```

另外，也不用创建所有的中间 group，可以直接指定完整路径：

```python
>>> dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i')
>>> dset3.name
'/subgroup2/dataset_three'
```

`Group` 支持大多数 Python dict 接口。如检索语法：

```python
dataset_three = f['subgroup2/dataset_three']
```

- 迭代 group 返回其成员名称

```python
>>> for name in f:
...     print(name)
mydataset
subgroup
subgroup2
```

- 测试是否包含指定名称

```python
>>> "mydataset" in f
True
>>> "somethingelse" in f
False
```

- 也可以使用完整路径名称

```python
>>> "subgroup/another_dataset" in f
True
```

另外还支持 `keys()`, `values()`, `items()`, `iter()` 以及 `get()`。

- 迭代 group 只访问其直接成员，要遍历整个文件，可以用 `Group` 的 `visit()` 和 `visititems()`

`visit()` 和 `visititems()` 的参数是 callable 对象。

```python
>>> def printname(name):
...     print(name)
>>> f.visit(printname)
mydataset
subgroup
subgroup/another_dataset
subgroup2
subgroup2/dataset_three
```

## 属性

HDF5 可以将元数据存储在它所描述的数据旁边。所有 group 和 dataset 都支持属性（attributes）。

属性通过 `attrs` 代理对象访问，同样实现了 dict 接口：

```python
>>> dset.attrs['temperature'] = 99.5
>>> dset.attrs['temperature']
99.5
>>> 'temperature' in dset.attrs
True
```

## 参考

- https://docs.h5py.org/en/stable/quick.html
