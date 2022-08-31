# 高级 API

- [高级 API](#高级-api)
  - [File Objects](#file-objects)
    - [打开或创建文件](#打开或创建文件)
    - [File drivers](#file-drivers)
    - [Python file-like objects](#python-file-like-objects)
    - [Version bounding](#version-bounding)
    - [Closing files](#closing-files)
    - [User block](#user-block)
    - [Filenames on different systems](#filenames-on-different-systems)
    - [Chunk cache](#chunk-cache)
  - [Groups](#groups)
  - [Datasets](#datasets)
    - [Chunked](#chunked)
  - [Filter](#filter)

## File Objects

文件对象是 HDF5 的入口。除了文件相关功能，每个文件对象也是 HDF5 文件的 root group。

### 打开或创建文件

```python
f = h5py.File('myfile.hdf5','r')
```

|模式|说明|
|---|---|
|r|Readonly, file must exist (default)|
|r+|Read/write, file must exist|
|w|Create file, truncate if exists|
|w- or x|Create file, fail if exists|
|a|Read/write if exists, create otherwise|

### File drivers

HDF5 提供各种不同的底层驱动，它们将逻辑 HDF5 地址空间映射到不同的存储机制。可以在打开文件时指定使用的驱动：

```python
f = h5py.File('myfile.hdf5', driver=<driver name>, <driver_kwds>)
```

例如，HDF5 "core" driver 可用于创建纯内存的 HDF5 文件，在关闭时可以选择写入 disk。下面是支持的 drivers。

- **None**

**Strongly recommended**. Use the standard HDF5 driver appropriate for the current platform. On UNIX, this is the H5FD_SEC2 driver; on Windows, it is H5FD_WINDOWS.

- ‘sec2’

Unbuffered, optimized I/O using standard POSIX functions.

- ‘stdio’

Buffered I/O using functions from stdio.h.

- ‘core’

Store and manipulate the data in memory, and optionally write it back out when the file is closed. Using this with an existing file and a reading mode will read the entire file into memory. Keywords:

**backing_store:**

If True (default), save changes to the real file at the specified path on close() or flush(). If False, any changes are discarded when the file is closed.

**block_size:**
Increment (in bytes) by which memory is extended. Default is 64k.

- ‘family’

Store the file on disk as a series of fixed-length chunks. Useful if the file system doesn’t allow large files. Note: the filename you provide must contain a printf-style integer format code (e.g. %d”), which will be replaced by the file sequence number. Keywords:

memb_size: Maximum file size (default is 2**31-1).

- ‘fileobj’

Store the data in a Python file-like object; see below. This is the default if a file-like object is passed to File.

- ‘split’

Splits the meta data and raw data into separate files. Keywords:

**meta_ext:**
Metadata filename extension. Default is ‘-m.h5’.

**raw_ext:**
Raw data filename extension. Default is ‘-r.h5’.

- ‘ros3’

Allows read only access to HDF5 files on S3. Keywords:

**aws_region:**
Name of the AWS “region” where the S3 bucket with the file is, e.g. b"us-east-1". Default is b''.

**secret_id:**
“Access ID” for the resource. Default is b''.

**secret_key:**
“Secret Access Key” associated with the ID and resource. Default is b''.

The argument values must be bytes objects.

### Python file-like objects

`File` 的第一个参数可能是 file-like 对象，如 `io.BytesIO` 或 `tempfile.TemporaryFile`。这是创建临时 HDF5 文件的便捷方法，可用于测试或通过网络发送中。

file-like 对象必须以 binary I/O 打开，且包含方法 `read()`, `write()`, `seek()`, `tell()`, `truncate()` 和 `flush()`。

```python
tf = tempfile.TemporaryFile()
f = h5py.File(tf, 'w')
```

在 file-like 对象关闭后访问 `File` 实例，结果不确定。

当使用内存对象如 `io.BytesIO` 时，写入的数据会占据内存空间。如果需要写入大量数据，建议使用 `tempfile` 中的函数将临时数据存储到 disk。

```python
"""Create an HDF5 file in memory and retrieve the raw bytes

This could be used, for instance, in a server producing small HDF5
files on demand.
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

> **[!WARNING]**
> 在 HDF5 中使用 file-like 对象时，确保先关闭 HDF5，再关闭文件对象。

> **[!WARNING]**
> 使用 file-like 对象时，使用服务线程实现 file-like API 可能导致进程死锁。
> `h5py` 通过一个 global lock 序列化对 low-level hdf5 函数的访问。当调用 file-like 方法删除、释放 `h5py` 对象时持有该锁。因此，如果在服务线程上触发循环垃圾收集，程序死锁。即服务线程在获得锁之前不能继续运行，持有锁的线程在服务线程完成前不会释放锁。
> 应该尽可能避免创建循环引用（通过 `weakrefs` 或手动终止循环）

### Version bounding

HDF5 默认以尽可能兼容的方式写入文件，以便旧版本仍然能够读取。但是，如果放弃一定程度的向后兼容性，则可能有性能优势。使用 `libver` 选项，可以指定最小和最大兼容程度：

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

### Closing files

调用 `File.close()` 或使用 `with h5py.File(...)` block 关闭文件后，文件相关的任何对象（groups, datasets）不可用。HDF5 称其为 'strong' closing。

如果文件对象超出 Python 代码的作用域，只有在没有属于该文件的其它对象时才会关闭该文件，HDF5 称其为 'weak' closing。

```python
with h5py.File('f1.h5', 'r') as f1:
    ds = f1['dataset']

# ERROR - can't access dataset, because f1 is closed:
ds[0]

def get_dataset():
    f2 = h5py.File('f2.h5', 'r')
    return f2['dataset']
ds = get_dataset()

# OK - f2 is out of scope, but the dataset reference keeps it open:
ds[0]

del ds  # Now f2.h5 will be closed
```

### User block

HDF5 允许用户在文件开头插入任何数据，这部分保留空间称为 *user block*。user block 的长度只能在创建文件时使用 `userblock_size` 参数指定。user block 长度可以是 0（默认），也可以是不小于 512 的 2 的指数。可以使用 `File.userblock_size` 属性查询。

不支持在打开的文件上修改 user block，这是 HDF5 库的一个限制。

但是，关闭文件后，只要不超过 user block 区域，就可以随意对文件开头进行读写。

### Filenames on different systems

不同操作系统使用不同的编码存储文件名。此外，Python 中至少有两种文件名表示形式，即编码为 `bytes` 或 Unicode string `str`。

h5py 的高级接口总是以 `str` 返回文件名，如 `File.filename`。h5py 输入文件名支持 `str` 和 `bytes`。大多时候首选 Unicode `str`，但也有一些注意事项。

> **[!NOTE]**
> HDF5 以 byte (C `char*`) 处理文件名，h5py low-level API 也是如此。

- macOS (OSX)

只接受 UTF-8 路径。

- Linux (non-macOS Unix)

Unix-like 系统使用 native bytes。

### Chunk cache

## Groups

Group 是 HDF5 文件使用的容器机制。从 Python 角度看，有点像 dict，"keys" 为 group 的名称，"values" 是其成员对象。

## Datasets

### Chunked

分块存储，数据集被分成规则大小的片段，随机存储在磁盘上，并使用 B 树进行索引。分块存储使得调整数据集大小很方便，并且由于数据存储在固定大小的分块中，所以可以使用压缩过滤器。

使用 `chunks` 关键字指定 chunk shape 并启用分块存储：

```python
dset = f.create_dataset("chunked", (1000, 1000), chunks=(100, 100))
```

数据将以 shape (100, 100) 的 block 进行读写；例如，数据 `dset[0:100,0:100]` 在文件中存储在一起。

Chunk 对性能有影响。建议将 chunk 大小保持在 10 kb 到 1MB 之间，数据集大的设置更大的 chunk。当访问 chunk 中的任意元素，整个 chunk 被读入内存。

因为 chunk shape 并不好选择，所以可以让 h5py 为了选择一个 shape：

```python
dset = f.create_dataset("autochunk", (1000, 1000), chunks=True)
```

当使用 compression 或 `maxshape` 且没有指定 chunk shape，auto-chunking 自动启用。

`iter_chunks` 可以迭代 chunk：

```python
for s in dset.iter_chunks():
     arr = dset[s]  # get numpy array for chunk
```

## Filter

- **Fletcher32 filter**

向每个 block 添加一个 checksum 以检测数据损坏。读取损坏的 chunk 会失败并抛出错误。该 filter 没有明显的速度损失。显然，它不应该与有损数据压缩 filter 一起使用。

将 `Group.create_dataset()` 的 `fletcher32` 设置为 True 启用该功能。

使用示例：

```python
"""
This example shows how to read and write data to a dataset using
the Fletcher32 checksum filter.
"""
import numpy as np
import h5py

FILE = "h5ex_d_checksum.h5"
DATASET = "DS1"

DIM0 = 32
DIM1 = 64
CHUNK0 = 4
CHUNK1 = 8


def run():
    # Initialize the data.
    wdata = np.zeros((DIM0, DIM1), dtype=np.int32)
    for i in range(DIM0):
        for j in range(DIM1):
            wdata[i][j] = i * j - j

    # Create the dataset with chunking and the Fletcher32 filter.
    with h5py.File(FILE, 'w') as f:
        dset = f.create_dataset(DATASET, (DIM0, DIM1), chunks=(CHUNK0, CHUNK1),
                                fletcher32=True, dtype='<i4')
        dset[...] = wdata

    with h5py.File(FILE) as f:
        dset = f[DATASET]
        if f[DATASET].fletcher32:
            print("Filter type is H5Z_FILTER_FLETCHER32.")
        else:
            raise RuntimeError("Fletcher32 filter not retrieved.")

        rdata = np.zeros((DIM0, DIM1))
        dset.read_direct(rdata)

    # Verify that the dataset was read correctly.
    np.testing.assert_array_equal(rdata, wdata)


if __name__ == "__main__":
    run()
```