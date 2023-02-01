# 快速入门

- [快速入门](#快速入门)
  - [安装](#安装)
  - [核心概念](#核心概念)
  - [创建文件](#创建文件)
  - [Group 和分层结构](#group-和分层结构)
  - [参考](#参考)

***

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

- `dset` 不是数组，而是 HDF5 的 `Dataset` 对象。和 NumPy 数组类似，dataset 也有 shape 和 datatype：

```python
>>> dset.shape
(100,)
>>> dset.dtype
dtype('int32')
```

- `Dataset` 支持切片，如读取和写入数据

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

文件支持上下文管理器，所以可以使用如下代码：

```python
import h5py
import numpy as np

with h5py.File("mytestfile.hdf5", "w") as f:
    dset = f.create_dataset("mydataset", (100,), dtype='i')
```

## Group 和分层结构

HDF 代表分层数据格式（Hierarchical Data Format）。HDF5 文件中的每个对象都有一个名称，以 posix 样式的分层结构存放，`/` 作为分隔符。例如：

```python
>>> dset.name
'/mydataset'
```

该系统中的文件夹称为 group。创建的 `File` 对象本身就是一个 group，称为根 group，名称为 `/`：

```python
>>> f.name
'/'
```



## 参考

- https://docs.h5py.org/en/stable/quick.html
