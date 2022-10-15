# Group

- [Group](#group)
  - [简介](#简介)
  - [创建 group](#创建-group)
  - [dict 接口和链接](#dict-接口和链接)
    - [硬链接](#硬链接)
    - [软链接](#软链接)
    - [外部链接](#外部链接)
  - [API](#api)
  - [Link 类](#link-类)
  - [参考](#参考)

Last updated: 2022-10-16, 00:33
@author Jiawei Mao
****

## 简介

Group 是 HDF5 文件的容器机制。从 Python 角度看，有点像 dict，"keys" 为 group 成员的名称，"values" 是成员对象本身（`Group` 和 `Dataset`）。

Group 对象包含 HDF5 的大部分功能。文件对象除了作为文件入口，也是 root group。

```python
>>> f = h5py.File('foo.hdf5', 'w')
>>> f.name
'/'
>>> list(f.keys())
[]
```

文件中所有对象的名称都是文本字符串（`str`）。在传入 HDF C 库前，将使用 UTF-8 对名称进行编码。也可以使用 byte string 检索对象，这类字符串将原样传入 HDF5。

## 创建 group

创建新的 group：

```python
>>> grp = f.create_group("bar")
>>> grp.name
'/bar'
>>> subgrp = grp.create_group("baz")
>>> subgrp.name
'/bar/baz'
```

也可以隐式创建多个中间 group：

```python
>>> grp2 = f.create_group("/some/long/path")
>>> grp2.name
'/some/long/path'
>>> grp3 = f['/some/long']
>>> grp3.name
'/some/long'
```

## dict 接口和链接

Group 实现了 Python dict 接口的部分方法。包含 `keys()`, `values` 等方法，并支持迭代。更重要的是，Group 支持索引语言和标准异常：

```python
>>> myds = subgrp['MyDS']
>>> missing = subgrp["missing"]
KeyError: "Unable to open object (object 'MyDS' doesn't exist)"
```

可以使用标准语法从文件中删除对象：

```python
del subgroup["MyDataset"]
```

> **Note:** 在 Python 3 中使用 h5py 时，`keys()`, `values` 和 `items` 方法返回 view-like 对象，而不是 list。这些对象支持成员测试和迭代，而不能像 list 那样切片。

group 中的对象默认按字母数字顺序迭代。但是，如果使用 `track_order=True` 创建 group，则 HDF5 文件会记住数据插入顺序，迭代按照插入顺序进行。后者与 Python 3.7+ dict 一致。

所有新创建 group 的 `track_order` 可以使用 `h5.get_config().track_order` 全局设置。

### 硬链接

将对象分配给 group 中的名称会发生什么？这取决于被分配的对象类型。对 NumPy 数组或其它数据，默认创建 HDF5 dataset：

```python
>>> grp["name"] = 42
>>> out = grp["name"]
>>> out
<HDF5 dataset "name": shape (), type "<i4">
```

当被存储的对象是已有的 Group 或 Dataset，则创建一个到该对象的 link：

```python
>>> grp["other name"] = out
>>> grp["other name"]
<HDF5 dataset "other name": shape (), type "<i4">
```

注意，这不是 dataset 的副本。就像 UNIX 文件系统的硬链接一样，HDF5 文件中的对象可以存储在多个 group 中：

```python
>>> grp["other name"] == grp["name"]
True
```

### 软链接

还是和 UNIX 文件系统一样，HDF5 group 可以包含软链接或符号链接，软连接包含文件路径而不是指向对象的指针。在 h5py 中使用 `h5py.SoftLink` 创建：

```python
>>> myfile = h5py.File('foo.hdf5', 'w')
>>> group = myfile.create_group("somegroup")
>>> myfile["alias"] = h5py.SoftLink('/somegroup')
```

如果链接指向的目标被移除，链接变为 "dangle"：

```python
>>> del myfile['somegroup']
>>> print(myfile['alias'])
KeyError: 'Component not found (Symbol table: Object not found)'
```

### 外部链接

HDF 1.8 新增的外部链接可以看做软连接的加强版，可以同时指定文件名和对象的路径。可以引用任何文件中的对象，语法和软链接类似：

```python
myfile = h5py.File('foo.hdf5', 'w')
myfile['ext link'] = h5py.ExternalLink("otherfile.hdf5", "/path/to/resource")
```

当访问该链接，将打开 "otherfile.hdf5" 文件，并返回位于 "/path/to/resource" 的对象。

由于检索到的对象位于不同文件，因此它的 `.file` 和 `.parent` 属性将引用该文件中的对象，而不是链接所在的文件。

> **Note:** 如果外部链接指向的文件已打开，则无法访问该链接。这与 HDF5 管理文件权限有关。

> **Note:** 文件名以 byte 形式存储在文件中，通常以 UTF-8 编码。大多时候都没问题，如果在一个平台创建的问题，在另一个平台访问，有可能出问题。

## API

```python
class h5py.Group(identifier)
```

Group 对象一般通过打开文件的对象或调用 `Group.create_group()` 创建。使用 `GroupID` 实例调用构造函数，创建绑定到已有 low-level 识别符的 group。

```python
__iter__()
```

迭代直接和 group 连接的对象的名称。使用 `Group.visit()` 或 `Group.visititems()` 递归访问 group 成员。

```python
__contains__(name)
```

类似 dict 的成员测试。`name` 可以是相对或绝对路径。

```python
__getitem__(name)
```

Retrieve an object. name may be a relative or absolute path, or an object or region reference. See Dict interface and links.

```python
__setitem__(name, value)
```

Create a new link, or automatically create a dataset. See Dict interface and links.

```python
__bool__()
```

Check that the group is accessible. A group could be inaccessible for several reasons. For instance, the group, or the file it belongs to, may have been closed elsewhere.

```python
>>> f = h5py.open(filename)
>>> group = f["MyGroup"]
>>> f.close()
>>> if group:
...     print("group is accessible")
... else:
...     print("group is inaccessible")
group is inaccessible
```

```python
keys()
```

Get the names of directly attached group members. Use Group.visit() or Group.visititems() for recursive access to group members.

Returns
set-like object.

```python
values()
```

Get the objects contained in the group (Group and Dataset instances). Broken soft or external links show up as None.

Returns
a collection or bag-like object.

items()¶
Get (name, value) pairs for object directly attached to this group. Values for broken soft or external links show up as None.

Returns
a set-like object.

```python
get(name, default=None, getclass=False, getlink=False)
```

检索 item 或 item 相关信息。`name` 和 `default` 参数与标准 Python `dict.get` 相同。

参数：

- `name` – 检索对象的名称，可以是相对或绝对路径。
- `default` – 如果没有找到对象，则返回此对象。
- `getclass` – `True` 时返回对象的类，即返回 `Group` 或 `Dataset`。
- `getlink` – `True` 时以 `HardLink`, `SoftLink` 或 `ExternalLink` 实例返回 link 类型。如果 `getclass=True`，则返回对应的链接类，不需要实例化。

visit(callable)¶
Recursively visit all objects in this group and subgroups. You supply a callable with the signature:

callable(name) -> None or return value
name will be the name of the object relative to the current group. Return None to continue visiting until all objects are exhausted. Returning anything else will immediately stop visiting and return that value from visit:

def find_foo(name):
    """ Find first object with 'foo' anywhere in the name """
    if 'foo' in name:
        return name
group.visit(find_foo)
'some/subgroup/foo'
visititems(callable)¶
Recursively visit all objects in this group and subgroups. Like Group.visit(), except your callable should have the signature:

callable(name, object) -> None or return value
In this case object will be a Group or Dataset instance.

move(source, dest)¶
Move an object or link in the file. If source is a hard link, this effectively renames the object. If a soft or external link, the link itself is moved.

Parameters
source (String) – Name of object or link to move.

dest (String) – New location for object or link.

copy(source, dest, name=None, shallow=False, expand_soft=False, expand_external=False, expand_refs=False, without_attrs=False)¶
Copy an object or group. The source can be a path, Group, Dataset, or Datatype object. The destination can be either a path or a Group object. The source and destination need not be in the same file.

If the source is a Group object, by default all objects within that group will be copied recursively.

When the destination is a Group object, by default the target will be created in that group with its current name (basename of obj.name). You can override that by setting “name” to a string.

Parameters
source – What to copy. May be a path in the file or a Group/Dataset object.

dest – Where to copy it. May be a path or Group object.

name – If the destination is a Group object, use this for the name of the copied object (default is basename).

shallow – Only copy immediate members of a group.

expand_soft – Expand soft links into new objects.

expand_external – Expand external links into new objects.

expand_refs – Copy objects which are pointed to by references.

without_attrs – Copy object(s) without copying HDF5 attributes.

create_group(name, track_order=None)¶
Create and return a new group in the file.

Parameters
name (String or None) – Name of group to create. May be an absolute or relative path. Provide None to create an anonymous group, to be linked into the file later.

track_order – Track dataset/group/attribute creation order under this group if True. Default is h5.get_config().track_order.

Returns
The new Group object.

require_group(name)¶
Open a group in the file, creating it if it doesn’t exist. TypeError is raised if a conflicting object already exists. Parameters as in Group.create_group().

```python
create_dataset(name, shape=None, dtype=None, data=None, **kwds)
```

创建新的数据集。选择说明可以参考 [Dataset](dataset.md)。

参数：

- `name` – 要创建数据集的名称。可以是相对或绝对路径。提供 `None` 创建匿名数据集，用于稍后链接到文件。
- `shape` – 新数据集的 shape (Tuple)。
- `dtype` – 新数据集的数据类型。
- `data` – Initialize dataset to this (NumPy array).

chunks – Chunk shape, or True to enable auto-chunking.

maxshape – Dataset will be resizable up to this shape (Tuple). Automatically enables chunking. Use None for the axes you want to be unlimited.

compression – Compression strategy. See Filter pipeline.

compression_opts – Parameters for compression filter.

scaleoffset – See Scale-Offset filter.

shuffle – Enable shuffle filter (T/F). See Shuffle filter.

fletcher32 – Enable Fletcher32 checksum (T/F). See Fletcher32 filter.

fillvalue – This value will be used when reading uninitialized parts of the dataset.

track_times – Enable dataset creation timestamps (T/F).

track_order – Track attribute creation order if True. Default is h5.get_config().track_order.

external – Store the dataset in one or more external, non-HDF5 files. This should be an iterable (such as a list) of tuples of (name, offset, size) to store data from offset to offset + size in the named file. Each name must be a str, bytes, or os.PathLike; each offset and size, an integer. The last file in the sequence may have size h5py.h5f.UNLIMITED to let it grow as needed. If only a name is given instead of an iterable of tuples, it is equivalent to [(name, 0, h5py.h5f.UNLIMITED)].

allow_unknown_filter – Do not check that the requested filter is available for use (T/F). This should only be set if you will write any data with write_direct_chunk, compressing the data before passing it to h5py.

require_dataset(name, shape=None, dtype=None, exact=None, **kwds)¶
Open a dataset, creating it if it doesn’t exist.

If keyword “exact” is False (default), an existing dataset must have the same shape and a conversion-compatible dtype to be returned. If True, the shape and dtype must match exactly.

Other dataset keywords (see create_dataset) may be provided, but are only used if a new dataset is to be created.

Raises TypeError if an incompatible object already exists, or if the shape or dtype don’t match according to the above rules.

Parameters
exact – Require shape and type to match exactly (T/F)

create_dataset_like(name, other, **kwds)¶
Create a dataset similar to other, much like numpy’s _like functions.

Parameters
name – Name of the dataset (absolute or relative). Provide None to make an anonymous dataset.

other – The dataset whom the new dataset should mimic. All properties, such as shape, dtype, chunking, … will be taken from it, but no data or attributes are being copied.

Any dataset keywords (see create_dataset) may be provided, including shape and dtype, in which case the provided values take precedence over those from other.

create_virtual_dataset(name, layout, fillvalue=None)¶
Create a new virtual dataset in this group. See Virtual Datasets (VDS) for more details.

Parameters
name (str) – Name of the dataset (absolute or relative).

layout (VirtualLayout) – Defines what source data fills which parts of the virtual dataset.

fillvalue – The value to use where there is no data.

build_virtual_dataset()¶
Assemble a virtual dataset in this group.

This is used as a context manager:

with f.build_virtual_dataset('virt', (10, 1000), np.uint32) as layout:
    layout[0] = h5py.VirtualSource('foo.h5', 'data', (1000,))
Inside the context, you populate a VirtualLayout object. The file is only modified when you leave the context, and if there’s no error.

Parameters
name (str) – Name of the dataset (absolute or relative)

shape (tuple) – Shape of the dataset

dtype – A numpy dtype for data read from the virtual dataset

maxshape (tuple) – Maximum dimensions if the dataset can grow (optional). Use None for unlimited dimensions.

fillvalue – The value used where no data is available.

attrs¶
Attributes for this group.

id¶
The groups’s low-level identifier; an instance of GroupID.

ref¶
An HDF5 object reference pointing to this group. See Using object references.

regionref¶
A proxy object allowing you to interrogate region references. See Using region references.

name¶
String giving the full path to this group.

file¶
File instance in which this group resides.

```python
parent
```

`Group` instance containing this group.

## Link 类

```python
class h5py.HardLink
```

用来支持 `Group.get()`。没有状态，也不提供任何属性或方法。

## 参考

- https://docs.h5py.org/en/stable/high/group.html
