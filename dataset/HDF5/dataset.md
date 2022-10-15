# Datasets

- [Datasets](#datasets)
  - [简介](#简介)
  - [创建数据集](#创建数据集)
  - [读写数据](#读写数据)
    - [多重索引](#多重索引)
    - [长度和迭代](#长度和迭代)
  - [分块存储](#分块存储)
  - [Resizable datasets](#resizable-datasets)
  - [Filter pipeline](#filter-pipeline)
    - [无损压缩过滤器](#无损压缩过滤器)
    - [自定义压缩过滤器](#自定义压缩过滤器)
    - [Scale-Offset 过滤器](#scale-offset-过滤器)
    - [shuffle 过滤器](#shuffle-过滤器)
    - [Fletcher32 过滤器](#fletcher32-过滤器)
  - [Multi-Block 选择](#multi-block-选择)
  - [Fancy indexing](#fancy-indexing)
  - [参考](#参考)

***

## 简介

Datasets 类似于 NumPy 数组，是数据元素逇齐次集合，数据类型不可变，矩形 shape。与 NumPy 数组不同的是，它支持各种存储特性，如压缩、错误检测和 chunked I/O。

Dataset 在 h5py 中以 `h5py.Dataset` 类表示，支持熟悉的 NumPy 操作，如切片和各种属性：

- `shape`
- `size`
- `ndim`
- `dtype`
- `nbytes`

h5py 支持大多数 NumPy dtypes，并使用相同的字符代码（如 `'f'`, `'i8'`）和 dtype 机制。h5py 支持的所有 dtypes 可参考 [FAQ](https://docs.h5py.org/en/stable/faq.html)。

## 创建数据集

新数据集使用 `Group.create_dataset()` 或 `Group.require_dataset()` 创建。已有数据集使用 group 索引语言检索 `dset = group["name"]`。

初始化数据集，需要指定其 name、shape，以及可选的数据类型（默认 `'f'`）：

```python
dset = f.create_dataset("default", (100,)) # name, shape
dset = f.create_dataset("ints", (100,), dtype='i8')
```

也可以通过 data 参数使用已有 NumPy 数组初始化数据集：

```python
arr = np.arange(100)
dset = f.create_dataset("init", data=arr)
```

关键字 `shape` 和 `dtype` 可以与 `data` 一起使用。设置后覆盖 `data.shape` 和 `data.dtype`，需要满足两个条件：

1. `shape` 中的数据点总数与 `data.shape` 的数据点总数匹配；
2. `data.dtype` 能转换为要求的 `dtype`。

## 读写数据

HDF5 数据集使用 NumPy 切片语法来读写文件。切片规范直接翻译为 HDF5 "hyperslab" 选择，是快速访问文件中数据的有效方式。支持以下切片参数：

- Indices: 可以转换为 Python long 的任意类型；
- Slices：例如 `[:]` 或 `[0:10]`
- Field names，对复合数据
- Empty tuple：用于检索所有数据或标量数据

例如：

```python
dset = f.create_dataset("MyDataset", (10, 10, 10), 'f')
dset[0, 0, 0]
dset[0, 2:10, 1:9:3]
dset[:, ::2, 5]
dset[0]
dset[1, 5]
dset[0, ...]
dset[..., 6]
dset[()]
```

对复合数据，建议将字段名和数字切片分开：

```python
dset.fields("FieldA")[:10]  # Read a single field
dset[:10]["FieldA"]  # Read all fields, select in NumPy
```

也可以混合使用索引和字段名，如 `dset[:10, "FieldA"]`，但是在 h5py 未来版本中可能移除。

对标量数据集，可以使用与 NumPy 相同的语法：`result = dset[()]`。换句话说，使用空 tuple 来检索。

对简单的切片，支持广播：

```python
dset[0,:,:] = np.arange(10)  # Broadcasts to (10,10)
```

广播使用重复 hyperslab 选择实现，对大的目标选择也没问题，但是只支持简单切片（integer, slice 以及省略号）。

### 多重索引

将 numpy 数组载入内存将索引数据集一次。如果尝试再次索引以写入数据，你会发现似乎没有任何变化：

```python
>>> f = h5py.File('my_hdf5_file.h5', 'w')
>>> dset = f.create_dataset("test", (2, 2))
>>> dset[0][1] = 3.0  # No effect!
>>> print(dset[0][1])
0.0
```

上面的赋值只修改了加载的数组，它等价于：

```python
>>> new_array = dset[0]
>>> new_array[1] = 3.0
>>> print(new_array[1])
3.0
>>> print(dset[0][1])
0.0
```

要写入数据集，将上述索引合并到一步：

```python
>>> dset[0, 1] = 3.0
>>> print(dset[0, 1])
3.0
```

### 长度和迭代

和 NumPy 数组一样，数据集的 `len()` 是第一个维度的长度，迭代数据集也是遍历第一个维度。然而，对 yield 的数据进行修改不影响文件中的数据。在迭代时 resize 数据集的效果没定义。

在 32-bit 平台上，如果第一个维度大于 2**32，调用 `len(dataset)` 会失败。对大型数据集建议使用 `Dataset.len()`。

## 分块存储

使用默认设置创建的 HDF5 数据集是连续的，换句话说，在磁盘上以传统的 C 顺序存储。也可以使用 HDF5 的**分块存储**（chunked storage）创建数据集：即将数据集分成规则大小的片段，随机存储在磁盘上，并使用 B 树进行索引。

分块存储使得调整数据集大小很方便，并且由于数据存储在固定大小的分块中，所以可以使用压缩过滤器。

使用 `chunks` 关键字指定块 shape 以启用分块存储：

```python
dset = f.create_dataset("chunked", (1000, 1000), chunks=(100, 100))
```

数据将以 shape (100, 100) 的 block 进行读写；例如，数据 `dset[0:100,0:100]` 在文件中存储在一起。

Chunk 对性能有影响。建议将 chunk 大小保持在 10 kb 到 1MB 之间，数据集越大 chunk 相应增大。当访问 chunk 中的任意元素，整个 chunk 被读入内存。

chunk shape 并不好选择，可以让 h5py 自动选择 shape：

```python
dset = f.create_dataset("autochunk", (1000, 1000), chunks=True)
```

当使用 compression 或 `maxshape` 且没有手动指定 chunk shape，auto-chunking 自动启用。

`iter_chunks` 可以迭代 chunk：

```python
for s in dset.iter_chunks():
     arr = dset[s]  # get numpy array for chunk
```

## Resizable datasets

在 HDF5 中，可以调用 `Dataset.resize()` 调整数据集大小。在创建数据集时可以用 maxshape 关键字指定最大大小：

```python
dset = f.create_dataset("resizable", (10, 10), maxshape=(500, 20))
```

任意或所有维度都可以使用 `None` 标记为无限制，无限制维度可以增加到 HDF5 的最大元素量 2**64：

```python
dset = f.create_dataset("unlimited", (10, 10), maxshape=(None, 10))
```

> **Note:** h5py 中对数组进行 resize 的行为与 NumPy 不同；收缩任意 axis，则缺失部分的数据被舍弃。而不像 NumPy 数组那样重新排列。

## Filter pipeline

Chunk 数据可以通过 HDF5 *filter pipeline* 进行转换。最常见的是应用透明压缩。数据在保存到磁盘的过程中被压缩，在读取时自动解压缩。使用特定的压缩 filter 创建数据后，可以像平常一样读写数据，不需要特殊步骤。

在 `Group.create_dataset()` 方法中使用 `compression` 关键字启用压缩：

```python
dset = f.create_dataset("zipped", (100, 100), compression="gzip")
```

各个过滤器的选项可以用 `compression_opts` 指定：

```python
dset = f.create_dataset("zipped_max", (100, 100),
                        compression="gzip",
                        compression_opts=9)
```

### 无损压缩过滤器

**GZIP filter (`"gzip"`)**

HDF5 自带过滤器，可移植性好，压缩性好，速度适中。`compression_opts` 设置压缩级别，可以是 0 到 9 之间的整数，默认 4.

**LZF filter (`"lzf"`)**

h5py 自带过滤器（C 源码也可以用），压缩性中等，非常快。无额外选项。

**SZIP filter (`"szip"`)**

NASA 社区中使用的受专利限制的过滤器。由于法律原因无法直接使用。

### 自定义压缩过滤器

除了上面列出的压缩过滤器，还可以由底层的 HDF5 库动态加载其它过滤器。将过滤器编号以参数 `compression` 传给 `Group.create_dataset()` 方法。然后将`compression_opts` 参数传递给该过滤器。

**[hdf5plugin](https://pypi.org/project/hdf5plugin/)**

Python 包，包含几个流行过滤器，如 Blosc, LZ4 和 ZFP 等。

**[HDF5 Filter Plugins](https://portal.hdfgroup.org/display/support/HDF5+Filter+Plugins)**

HDF Group 提供的可直接下载的过滤器集合。

**[Registered filter plugins](https://portal.hdfgroup.org/display/support/Filters)**

公开的过滤器插件索引。

> **Note:** 压缩过滤器的底层实现会设置 `H5Z_FLAG_OPTIONAL` flag。表示如果压缩过滤器在写入时不压缩 block，不会抛出错误。当随后读取 block 时，过滤器被跳过。

### Scale-Offset 过滤器

使用 `compression` 关键字启用的过滤器是无损的，数据集输出的和输出完全相同。HDF5 还包含有损过滤器，用精度换取存储空间。

Scale-Offset 过滤器只适用于整数和浮点数。通过将 `Group.create_dataset()` 的 `scaleoffset` 设置为 integer 来启用 scale-offset filter。

对整数数据，它指定要保留的位数，设置 0 表示由 HDF5 自动计算 block 无损压缩所需的位数。对浮点数，标志小数点后要保留的位数。

> **!WARNING** 目前 scale-offset filter 不保留特殊的浮点数，即 NaN，inf。详情可参考 https://forum.hdfgroup.org/t/scale-offset-filter-and-special-float-values-nan-infinity/3379 。

### shuffle 过滤器

像 GZIP 或 LZF 这样面向 block 的压缩器对具有相似值的数据集效果很好。启用 shuffle 过滤器可以重排 block 中的字节，从而提高压缩比。没有明显的速度损失，无损压缩。

通过将 `Group.create_dataset()` 的 `shuffle` 设置为 True 来启用。

### Fletcher32 过滤器

向每个 block 添加校验和以检测数据是否损坏。读取损坏的 block 会抛出错误。没有明显的速度损失。显然，该过滤器不能和有损压缩过滤器一起使用。

通过将 `Group.create_dataset()` 的 `fletcher32` 设置为 True 来启用。

## Multi-Block 选择

完整的 H5Sselect_hyperslab API 通过 `MultiBlockSlice` 对象公开。`MultiBlockSlice` 需要四个元素来定义 (start, count, stride 和 block)，不像内置的切片，只需要三个元素。`MultiBlockSlice` 可以代替切片来选择由步长分隔的多个 block，而不是切片这种由 step 分隔的多个元素。

要理解这种切片原理，可参考 [HDF5 文档](https://support.hdfgroup.org/HDF5/Tutor/selectsimple.html)。

例如：

```python
>>> dset[...]
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
>>> dset[MultiBlockSlice(start=1, count=3, stride=4, block=2)]
array([ 1,  2,  5,  6,  9, 10])
```

它们可以在多维切片中与任何切片对象一起使用，包括其它 `MultiBlockSlice`。

## Fancy indexing

部分支持 NumPy 花式索引（fancy-indexing）语法。不过要谨慎使用，因为 HDF5 的底层机制的性能可能与预期不符。

对任意 axis，可以显式提供所需数据点列表，如：

```python
>>> dset.shape
(10, 10)
>>> result = dset[0, [1,3,8]]
>>> result.shape
(3,)
>>> result = dset[1:6, [5,8,9]]
>>> result.shape
(5, 3)
```

存在如下限制：

- 选择坐标必须按递增顺序给出；
- 

## 参考

- https://docs.h5py.org/en/stable/high/dataset.html
