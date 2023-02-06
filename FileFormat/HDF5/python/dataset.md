# Dataset

- [Dataset](#dataset)
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
  - [创建和读取空 datasets 和 attributes](#创建和读取空-datasets-和-attributes)
  - [h5py.Dataset](#h5pydataset)
  - [参考](#参考)

Last updated: 2022-10-16, 01:32
****

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
- 重复的选择被忽略；
- 非常长的列表（>1000 个元素）会导致性能较差。

也可以使用 NumPy boolean mask 数组来选择，返回的结果是按照标准 NumPy（C 样式）顺序排列的一维数组元素。底层实现会生成 一个待选择数据点的 list，因此在使用大型 mask 时要小新：

```python
>>> arr = numpy.arange(100).reshape((10,10))
>>> dset = f.create_dataset("MyDataset", data=arr)
>>> result = dset[arr > 50]
>>> result.shape
(49,)
```

版本 2.10 修改：允许使用空 list 进行选择。将返回相关维度中长度为 0 的数组。

## 创建和读取空 datasets 和 attributes

HDF5 具有 empty 或 null dataset 和 attribute 的概念。不同于 shape 为 `()` 的数组，也不同于 HDF5 的标量数据空间，它具有关联类型，但没有 data 和 shape 的数据集。在 h5py 中，将其表示为 shape 为 `None` 的 dataset，或 `h5py.Empty` 实例。不能对空 dataset 和 attribute 进行切片。

- 使用 `h5py.Empty` 创建空属性：

```python
obj.attrs["EmptyAttr"] = h5py.Empty("f")
```

- 读取空属性返回 `h5py.Empty`：

```python
>>> obj.attrs["EmptyAttr"]
h5py.Empty(dtype="f")
```

- 在 `create_dataset` 中指定 `dtype` 但指定 `shape` 来创建空 dataset:

```python
grp.create_dataset("EmptyDataset", dtype="f")
```

- 或者将 `data` 参数设置为 `h5py.Empty` 实例：

```python
grp.create_dataset("EmptyDataset", data=h5py.Empty("f"))
```

空 dataset 的 shape 为 `None`，这是确定数据集是否为空的最佳方法。空 dataset 可以以类似标量 dataset 的方式读取，即，如果 `empty_dataset` 是空 dataset:

```python
>>> empty_dataset[()]
h5py.Empty(dtype="f")
```

数据集的 dtype 可以通过 `<dset>.dtype` 访问。空数据集不能切片，且在空数据集上调用数据集的部分方法，如 `read_direct` 会抛出 `TypeError`。 

## h5py.Dataset

```python
class h5py.Dataset(identifier)
```

Dataset objects are typically created via Group.create_dataset(), or by retrieving existing datasets from a file. Call this constructor to create a new Dataset bound to an existing DatasetID identifier.

__getitem__(args)¶
NumPy-style slicing to retrieve data. See Reading & writing data.

__setitem__(args)¶
NumPy-style slicing to write data. See Reading & writing data.

__bool__()¶
Check that the dataset is accessible. A dataset could be inaccessible for several reasons. For instance, the dataset, or the file it belongs to, may have been closed elsewhere.

f = h5py.open(filename)
dset = f["MyDS"]
f.close()
if dset:
    print("datset accessible")
else:
    print("dataset inaccessible")
dataset inaccessible
read_direct(array, source_sel=None, dest_sel=None)¶
Read from an HDF5 dataset directly into a NumPy array, which can avoid making an intermediate copy as happens with slicing. The destination array must be C-contiguous and writable, and must have a datatype to which the source data may be cast. Data type conversion will be carried out on the fly by HDF5.

source_sel and dest_sel indicate the range of points in the dataset and destination array respectively. Use the output of numpy.s_[args]:

dset = f.create_dataset("dset", (100,), dtype='int64')
arr = np.zeros((100,), dtype='int32')
dset.read_direct(arr, np.s_[0:10], np.s_[50:60])
write_direct(source, source_sel=None, dest_sel=None)¶
Write data directly to HDF5 from a NumPy array. The source array must be C-contiguous. Selections must be the output of numpy.s_[<args>]. Broadcasting is supported for simple indexing.

astype(dtype)¶
Return a wrapper allowing you to read data as a particular type. Conversion is handled by HDF5 directly, on the fly:

dset = f.create_dataset("bigint", (1000,), dtype='int64')
out = dset.astype('int16')[:]
out.dtype
dtype('int16')
Changed in version 3.0: Allowed reading through the wrapper object. In earlier versions, astype() had to be used as a context manager:

with dset.astype('int16'):
    out = dset[:]
asstr(encoding=None, errors='strict')¶
Only for string datasets. Returns a wrapper to read data as Python string objects:

s = dataset.asstr()[0]
encoding and errors work like bytes.decode(), but the default encoding is defined by the datatype - ASCII or UTF-8. This is not guaranteed to be correct.

New in version 3.0.

fields(names)¶
Get a wrapper to read a subset of fields from a compound data type:

2d_coords = dataset.fields(['x', 'y'])[:]
If names is a string, a single field is extracted, and the resulting arrays will have that dtype. Otherwise, it should be an iterable, and the read data will have a compound dtype.

New in version 3.0.

iter_chunks()¶
Iterate over chunks in a chunked dataset. The optional sel argument is a slice or tuple of slices that defines the region to be used. If not set, the entire dataspace will be used for the iterator.

For each chunk within the given region, the iterator yields a tuple of slices that gives the intersection of the given chunk with the selection area. This can be used to read or write data in that chunk.

A TypeError will be raised if the dataset is not chunked.

A ValueError will be raised if the selection region is invalid.

New in version 3.0.

resize(size, axis=None)¶
Change the shape of a dataset. size may be a tuple giving the new dataset shape, or an integer giving the new length of the specified axis.

Datasets may be resized only up to Dataset.maxshape.

len()¶
Return the size of the first axis.

make_scale(name='')¶
Make this dataset an HDF5 dimension scale.

You can then attach it to dimensions of other datasets like this:

other_ds.dims[0].attach_scale(ds)
You can optionally pass a name to associate with this scale.

virtual_sources()¶
If this dataset is a virtual dataset, return a list of named tuples: (vspace, file_name, dset_name, src_space), describing which parts of the dataset map to which source datasets. The two ‘space’ members are low-level SpaceID objects.

shape¶
NumPy-style shape tuple giving dataset dimensions.

dtype¶
NumPy dtype object giving the dataset’s type.

size¶
Integer giving the total number of elements in the dataset.

nbytes¶
Integer giving the total number of bytes required to load the full dataset into RAM (i.e. dset[()]). This may not be the amount of disk space occupied by the dataset, as datasets may be compressed when written or only partly filled with data. This value also does not include the array overhead, as it only describes the size of the data itself. Thus the real amount of RAM occupied by this dataset may be slightly greater.

New in version 3.0.

ndim¶
Integer giving the total number of dimensions in the dataset.

maxshape¶
NumPy-style shape tuple indicating the maximum dimensions up to which the dataset may be resized. Axes with None are unlimited.

chunks¶
Tuple giving the chunk shape, or None if chunked storage is not used. See Chunked storage.

compression¶
String with the currently applied compression filter, or None if compression is not enabled for this dataset. See Filter pipeline.

compression_opts¶
Options for the compression filter. See Filter pipeline.

scaleoffset¶
Setting for the HDF5 scale-offset filter (integer), or None if scale-offset compression is not used for this dataset. See Scale-Offset filter.

shuffle¶
Whether the shuffle filter is applied (T/F). See Shuffle filter.

fletcher32¶
Whether Fletcher32 checksumming is enabled (T/F). See Fletcher32 filter.

fillvalue¶
Value used when reading uninitialized portions of the dataset, or None if no fill value has been defined, in which case HDF5 will use a type-appropriate default value. Can’t be changed after the dataset is created.

external¶
If this dataset is stored in one or more external files, this is a list of 3-tuples, like the external= parameter to Group.create_dataset(). Otherwise, it is None.

is_virtual¶
True if this dataset is a virtual dataset, otherwise False.

dims¶
Access to Dimension Scales.

is_scale¶
Return True if the dataset is also a dimension scale, False otherwise.

- **attrs**

该数据集的 Attributes.

id¶
The dataset’s low-level identifier; an instance of DatasetID.

ref¶
An HDF5 object reference pointing to this dataset. See Using object references.

regionref¶
Proxy object for creating HDF5 region references. See Using region references.

name¶
String giving the full path to this dataset.

file¶
File instance in which this dataset resides

parent¶
Group instance containing this dataset.

## 参考

- https://docs.h5py.org/en/stable/high/dataset.html
