# 文件对象

- [文件对象](#文件对象)
  - [简介](#简介)
  - [Opening \& creating files](#opening--creating-files)
  - [File drivers](#file-drivers)
  - [Python 文件对象](#python-文件对象)
  - [版本问题](#版本问题)
  - [关闭文件](#关闭文件)
  - [User block](#user-block)
  - [不同系统的文件名](#不同系统的文件名)
  - [Chunk cache](#chunk-cache)
  - [数据对齐](#数据对齐)
  - [Meta block size](#meta-block-size)
  - [h5py.File](#h5pyfile)
  - [参考](#参考)

Last updated: 2023-02-06, 16:37
****

## 简介

`File` 对象是 HDF5 文件访问的入口。除了下面列出的特定于 `File` 的功能，`File` 还是 HDF5 group，表示 HDF5 文件的 root group。

## Opening & creating files

HDF5 文件的工作方式与 Python 文件类似，支持多个标准模式，如 r/w/a，在不使用时也需要关闭。不过没有 "text" 和 "binary" 模式的概念。

```python
f = h5py.File('myfile.hdf5','r')
```

文件名为 byte string 或 unicode string。支持的模式包括：

| 模式 | 说明 |
|---|---|
| r | Readonly, file must exist (default) |
| r+ | Read/write, file must exist |
| w | Create file, truncate if exists |
| w- or x | Create file, fail if exists |
| a | Read/write if exists, create otherwise |

## File drivers

HDF5 提供各种不同的底层驱动，将 HDF5 逻辑地址空间映射到不同的存储机制。在打开文件时指定驱动：

```python
f = h5py.File('myfile.hdf5', driver=<driver name>, <driver_kwds>)
```

例如，HDF5 "core" driver 用于创建纯内存中的 HDF5 文件，在关闭时可以写入 disk。下面是 HDF5 支持的 drivers。

- **None**

**推荐**，选择适合当前平台的标准 HDF5 驱动。在 UNIX 上为 `H5FD_SEC2`，在 Windows 为 `H5FD_WINDOWS`。

- **‘sec2’**

使用标准 POSIX 函数实现的无缓存 I/O。

- **‘stdio’**

使用 stdio.h 函数实现的缓存 I/O。

- **‘core’**

在内存中存储和操作数据，在关闭文件时可以将内容写入 disk。以该 driver 读取 hdf5 文件，会将所有内容读入内存。关键字：
  
`backing_store`

`True` (默认) 表示在 `close()` 或 `flush()` 时将修改保存到指定路径的文件。`False` 表示关闭文件时丢弃更改。

`block_size`

每次扩展内存的大小（bytes），默认 64k。

- **‘family’**

将文件作为一系列固定长度的 chunk 存储在磁盘。当文件系统不允许大文件时非常有用。注意：提供的文件名必须包含 `printf` 风格的整数格式diamante，如 `%d`，它会被文件序列号取代。关键字：

`memb_size`

最大文件大小，默认 2**31-1.

- **‘fileobj’**

将数据存储在 Python 文件对象。当将文件对象传入 `File` 时，这是默认选择。

- **‘split’**

将元数据和原始数据存储到单独文件。关键字：

`meta_ext`

元数据文件名扩展，默认 ‘-m.h5’。

`raw_ext`

原始数据文件名扩展，默认 ‘-r.h5’。

- **‘ros3’**

允许 AWS S3 或 S3 兼容对象存储只读访问 HDF5 文件。HDF5 文件名必须是 `http://`, `https://` 或 `s3://` 资源位置的一个。其中 `s3://` 会被转换为 AWS [路径样式](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html#path-style-access)位置。关键字：

`aws_region`

存放文件的 S3 bucket 的 AWS 区域，例如 `b"us-east-1"`。默认 `b''`，需要 s3:// 位置。

`secret_id`

AWS 访问 key ID。默认 `b''`。

`secret_key`

AWS 秘密访问 key。默认 `b''`。

这三个参数值必须是 `bytes` 类型，用来激活 AWS 验证。

## Python 文件对象

`File` 的第一个参数是 file-like 对象，如 `io.BytesIO` 或 `tempfile.TemporaryFile`。这是创建临时 HDF5 文件的便捷方法，可用于测试或通过网络发送。

file-like 对象需以 binary 模式打开，且包含方法 `read()` (或 `readinto()`), `write()`, `seek()`, `tell()`, `truncate()` 和 `flush()`。

```python
tf = tempfile.TemporaryFile()
f = h5py.File(tf, 'w')
```

不支持在 file-like 对象关闭后访问 `File` 实例。

当使用内存对象如 `io.BytesIO`，写入的数据会占内存。如果需要写入大量数据，建议使用 `tempfile` 将数据写入临时文件。

```python
"""
在内存中创建 HDF5 文件，并检索 raw bytes.

可用在按需生成小型 HDF5 文件的服务器
"""
import io
import h5py

bio = io.BytesIO()
with h5py.File(bio, 'w') as f:
    f['dataset'] = range(10)

data = bio.getvalue() # data is a regular Python bytes object.
print("Total size:", len(data))
print("First bytes:", data[:10])
```

```txt
Total size: 1440
First bytes: b'\x89HDF\r\n\x1a\n\x00\x00'
```

> **WARNING**
> 在 HDF5 中使用 file-like 对象时，确保先关闭 HDF5，再关闭文件对象。

> **WARNING**
> 使用 file-like 对象时，使用服务线程实现 file-like API 可能导致进程死锁。
> 
> `h5py` 通过全局锁序列化访问 HDF5 底层函数。当调用 file-like 方法执行删除或释放 `h5py` 对象时持有该锁。因此，如果在服务线程上触发循环垃圾收集，程序死锁。即服务线程在获得锁之前不能继续运行，持有锁的线程在服务线程完成前不会释放锁。
> 应该尽可能避免创建循环引用（通过 `weakrefs` 或手动终止循环）。如果无法实现，在合适的线程手动出发垃圾回收或暂时禁用垃圾回收也可能有用。

## 版本问题

HDF5 默认以尽可能兼容的方式写入文件，以便旧版本仍然能够读取。但是，如果放弃一定程度的向后兼容性，则可能提高性能。使用 `File` 的 `libver` 选项可以指定最小和最大兼容程度：

```python
f = h5py.File('name.hdf5', libver='earliest') # most compatible
f = h5py.File('name.hdf5', libver='latest')   # most modern
```

默认 'earliest'。

HDF5 v1.10.2 之后，多了两个新的兼容级别：v108 (HDF5 1.8) 和 v110 (HDF5 1.10)，可以按如下方式指定版本：

```python
f = h5py.File('name.hdf5', libver=('earliest', 'v108'))
```

表示向后完全兼容到 HDF5 1.8。使用任何新的 HDF5 特征会抛出错误。

## 关闭文件

调用 `File.close()` 或使用 `with h5py.File(...)` 关闭文件后，文件相关的对象（groups, datasets）将不可用。HDF5 称其为强关闭（strong）。

当文件对象超出 Python 代码的作用域，只有在没有属于该文件的其它对象时才会关闭该文件，HDF5 称其为弱关闭（weak）。

```python
with h5py.File('f1.h5', 'r') as f1:
    ds = f1['dataset']

# ERROR - f1 关闭，无法访问 dataset
ds[0]

# 没有关闭文件
def get_dataset():
    f2 = h5py.File('f2.h5', 'r')
    return f2['dataset']

ds = get_dataset()

# OK - 虽然 f2 超出 scope，但因为 f2['dataset'] 持有其引用，使其没有关闭
ds[0]

del ds  # Now f2.h5 will be closed
```

## User block

HDF5 允许用户在文件开头插入任何数据，这部分保留空间称为 *user block*。

user block 的长度只能在创建文件时使用 `userblock_size` 参数指定，可以是 0（默认），也可以是不小于 512 的 2 的指数。使用 `File.userblock_size` 属性查询 block 大小。

不能在打开的文件上修改 user block，这是 HDF5 库的一个限制。

关闭文件后，只要不超过 user block 区域，就可以随意对文件开头进行读写。

## 不同系统的文件名

不同操作系统（以及不同文件系统）使用不同的编码存储文件名。此外，Python 中至少有两种文件名表示形式，即编码为 `bytes` 或 Unicode string `str`。

h5py 的高级接口以 `str` 返回文件名，如 `File.filename`。h5py 输入文件名支持 `str` 和 `bytes`。大多时候首选 Unicode `str`，但也有一些注意事项。

> **NOTE**
> HDF5 以 byte (C `char*`) 处理文件名，h5py low-level API 也是如此。

- **macOS (OSX)**

macOS 只接受 UTF-8 路径，所有最容易处理。

- **Linux (non-macOS Unix)**

Unix 系统使用 native bytes 为文件名，在本地编码和 unicode 之间进行转换。大多数现代系统默认使用 UTF-8，特别是从 Python 3.7 开始。

大多数情况使用 Unicode 没问题，如果预期的文件名编码不对，如在网络文件系统或可移动设备上，此时可能需要转换为 byte 来处理。

- **Windows**

Windows 系统以 Unicode 处理文件名，并且从 HDF5 1.10.6 开始，所有文件名都以 UTF-8 处理。

## Chunk cache

分块存储（Chunked storage）可以将数据集拆分成多块单独存储。当需要某个 chunk 中的某些数据，会将整个 chunk 读入内存。如果查询的数据所属 chunk 已在内存，就不用再次读取文件。数据集 chunk 的详细信息在创建数据集时设置，在打开文件时可以调整 chunk 的缓存行为。

设置 chunk 缓存的参数以 `rdcc` (raw data chunk cache) 开头。

- **rdcc_nbytes** 

`rdcc_nbytes` 设置每个数据集的原始缓存 chunk 总大小，默认 1 MB。为 chunk 大小乘以需要缓存的 chunk 数。

- **rdcc_w0** 

`rdcc_w0` 设置从缓存中删除 chunk 的策略。0 表示优先回收缓存的最近使用最少的 chunk；1 表示优先回收缓存的最近使用最少且已完全读取或写入的 chunk，如果没有完全读写的 chunk，则回收最近最少使用的 chunk。对 0 到 1 之间的值，则为两者的混合。因此，对需要多次访问的数据，应将其设置为接近 0，否则设置为接近 1。

- **rdcc_nslots**

`rdcc_nslots` 设置缓存中每个数据集的 chunk 槽数。为了在缓存中快速查找 chunk，为每个 chunk 分配了一个 unique hash 值。缓存包含指向所有 chunk 的指针的数组，称为哈希表。chunk 的 hash 值其实就是该 chunk 的指针在哈希表中的索引。

哈希表中的指针可能指向其它 chunk 或没指向任何内容，但哈希表的其它位置肯定没有指向该 chunk 的指针。因此通过检查哈希表中对应位置，就能判断对应 chunk 是否在缓存中。因此，如果两个或多个 chunk 共享相同的哈希值，那么这些 chunk 同时只能有一个在缓存中。因此确定哈希表的大小很重要（由 `rdcc_nslots` 设置）。根据哈希策略，该值取质数最佳。根据经验，该值至少是 `rdcc_nbytes` 能容纳的 chunk 数的 10 倍。为了获得最佳性能，将其设置为 chunk 数的 100 倍。默认 521.

详情可参考 [Chunking in HDF5](https://portal.hdfgroup.org/display/HDF5/Chunking+in+HDF5)。

## 数据对齐

在文件中创建数据集时，在文件内对齐 offset 是有利的。当数据对底层硬件对齐，可以优化读写时间，或有助于基于 MPI 的并行。

但是，将小变量与大的 block 对齐会导致文件中留下大片空白。因此，HDF5 对优化文件内数据对齐的方式只给出了两个选项，`File` 构造函数的两个参数 `alignment_threshold` 和 `alignment_interval` 分别设置数据对齐生效的阈值（bytes）和文件中对齐 bytes。

详情可参考 [H5P_SET_ALIGNMENT](https://portal.hdfgroup.org/display/HDF5/H5P_SET_ALIGNMENT)。

## Meta block size

元数据的空间在 HDF5 文件中以 block 的形式分配。`File` 构造函数的 `meta_block_size` 参数设置这些 block 的最小大小。设置较大的值可以将元数据合并到少数几个区域。设置较小的值可以减少整个文件的大小，特别是与 `libver` 选项结合使用时。 

更多信息可参考 HDF5 文档 [H5P_SET_META_BLOCK_SIZE](https://portal.hdfgroup.org/display/HDF5/H5P_SET_META_BLOCK_SIZE)。

## h5py.File

> **Note:** 与 Python 文件对象不同，`File.name` 属性给出的是 root group `/`。访问磁盘上的名称，使用 `File.filename`。

```python
classh5py.File(name, 
    mode='r', 
    driver=None, 
    libver=None, 
    userblock_size=None, 
    swmr=False, 
    rdcc_nslots=None, 
    rdcc_nbytes=None, 
    rdcc_w0=None, 
    track_order=None, 
    fs_strategy=None, 
    fs_persist=False, 
    fs_threshold=1, 
    fs_page_size=None, 
    page_buf_size=None, 
    min_meta_keep=0, 
    min_raw_keep=0, 
    locking=None, 
    alignment_threshold=1, 
    alignment_interval=1, 
    **kwds)
```

打开或创建文件。

注意：除了下面列出的特定于 `File` 的属性和方法，`File` 还继承了 `Group`。

**参数：**

- **name** – Name of file (bytes or str), or an instance of h5f.FileID to bind to an existing file identifier, or a file-like object (see Python file-like objects).

mode – Mode in which to open file; one of (“w”, “r”, “r+”, “a”, “w-“). See Opening & creating files.

driver – File driver to use; see File drivers.

libver – Compatibility bounds; see Version bounding.

userblock_size – Size (in bytes) of the user block. If nonzero, must be a power of 2 and at least 512. See User block.

- **swmr**

以 single-writer-multiple-reader 模式打开文件。仅当 mode='r' 时使用。

rdcc_nbytes – Total size of the raw data chunk cache in bytes. The default size is 
 (1 MiB) per dataset.

rdcc_w0 – Chunk preemption policy for all datasets. Default value is 0.75.

rdcc_nslots – Number of chunk slots in the raw data chunk cache for this file. Default value is 521.

track_order – Track dataset/group/attribute creation order under root group if True. Default is h5.get_config().track_order.

fs_strategy – The file space handling strategy to be used. Only allowed when creating a new file. One of “fsm”, “page”, “aggregate”, “none”, or None (to use the HDF5 default).

fs_persist – A boolean to indicate whether free space should be persistent or not. Only allowed when creating a new file. The default is False.

fs_page_size – File space page size in bytes. Only use when fs_strategy=”page”. If None use the HDF5 default (4096 bytes).

fs_threshold – The smallest free-space section size that the free space manager will track. Only allowed when creating a new file. The default is 1.

page_buf_size – Page buffer size in bytes. Only allowed for HDF5 files created with fs_strategy=”page”. Must be a power of two value and greater or equal than the file space page size when creating the file. It is not used by default.

min_meta_keep – Minimum percentage of metadata to keep in the page buffer before allowing pages containing metadata to be evicted. Applicable only if page_buf_size is set. Default value is zero.

min_raw_keep – Minimum percentage of raw data to keep in the page buffer before allowing pages containing raw data to be evicted. Applicable only if page_buf_size is set. Default value is zero.

locking –

The file locking behavior. One of:

False (or “false”) – Disable file locking

True (or “true”) – Enable file locking

”best-effort” – Enable file locking but ignore some errors

None – Use HDF5 defaults

Warning

The HDF5_USE_FILE_LOCKING environment variable can override this parameter.

Only available with HDF5 >= 1.12.1 or 1.10.x >= 1.10.7.

alignment_threshold – Together with alignment_interval, this property ensures that any file object greater than or equal in size to the alignement threshold (in bytes) will be aligned on an address which is a multiple of alignment interval.

alignment_interval – This property should be used in conjunction with alignment_threshold. See the description above. For more details, see Data alignment.

meta_block_size – Determines the current minimum size, in bytes, of new metadata block allocations. See Meta block size.

kwds – Driver-specific keywords; see File drivers.

**方法：**

- `__bool__()`

检查文件描述符是否有效，文件是否打开：

```python
>>> f = h5py.File(filename)
>>> f.close()
>>> if f:
...     print("file is open")
... else:
...     print("file is closed")
file is closed
```

- **close()**

关闭该文件。所有打开的对象将无效。

- **flush()**

将缓冲区刷新到磁盘。

- **id**

底层识别符（`FileID` 实例）。

- **filename**

文件在 disk 上的名称，Unicode 字符串。

- **mode**

字符串，文件打开模式：只读（"r"）或读写（"r+"）。无论以哪种模式打开文件，总是这两个值之一。

- **swmr_mode**

以 Single Writer Multiple Reader (SWMR) 模式访问文件时为 `True`。使用 `mode` 来区分读还是写 模式。

- **driver**

打开文件的驱动名称（string）。

- **libver**

包含版本设置的 2-tuple。参考[版本问题](#版本问题)

- **userblock_size**

user block 尺寸（bytes），一般为 0.

- **meta_block_size**

分配的元数据 block 最小尺寸（bytes），默认 2048.

## 参考

- https://docs.h5py.org/en/stable/high/file.html
