# lmdb

- [lmdb](#lmdb)
  - [简介](#简介)
  - [安装：Windows](#安装windows)
  - [命名数据库](#命名数据库)
  - [存储效率和限制](#存储效率和限制)
  - [内存使用](#内存使用)
  - [Bytestrings](#bytestrings)
  - [Buffers](#buffers)
  - [writemap 模式](#writemap-模式)
  - [资源管理](#资源管理)
  - [使用](#使用)
    - [创建 Environment](#创建-environment)
    - [修改数据库内容](#修改数据库内容)
    - [查询数据库](#查询数据库)
    - [完整演示](#完整演示)
  - [API](#api)
    - [lmdb.open](#lmdbopen)
    - [lmdb.version](#lmdbversion)
    - [Environment](#environment)
    - [Transaction](#transaction)
    - [Cursor](#cursor)
  - [参考](#参考)

Last updated: 2023-01-28, 13:57
****

## 简介

LMDB 全称为 Lightning Memory-Mapped Database，就是非常快的内存映射型数据库，LMDB使用内存映射文件，可以提供更好的输入/输出性能，对于用于神经网络的大型数据集( 比如 ImageNet )，可以将其存储在 LMDB 中。

LMDB效率高的一个关键原因是它是基于内存映射的，这意味着它返回指向键和值的内存地址的指针，而不需要像大多数其他数据库那样复制内存中的任何内容。

用LMDB数据库来存放图像数据，而不是直接读取原始图像数据的原因：

- 数据类型多种多样，比如：二进制文件、文本文件、编码后的图像文件jpeg、png等，不可能用一套代码实现所有类型的输入数据读取，因此通过LMDB数据库，转换为统一数据格式可以简化数据读取层的实现。
- lmdb具有极高的存取速度，大大减少了系统访问大量小文件时的磁盘IO的时间开销。LMDB将整个数据集都放在一个文件里，避免了文件系统寻址的开销，你的存储介质有多快，就能访问多快，不会因为文件多而导致时间长。LMDB使用了内存映射的方式访问文件，这使得文件内寻址的开销大幅度降低。

## 安装：Windows

```powershell
pip install lmdb
```

## 命名数据库

命名数据库要求在调用 `lmdb.open()` 或 `lmdb.Environment` 时提供 `max_dbs=` 参数。必须由打开环境的第一个进程或线程来完成。

创建好 `Environment` 后，可以使用 `Environment.open_db()` 创建新的命名数据库。

## 存储效率和限制

数据被分到与操作系统的 VM 页面大小相匹配的 page 中，通常为 4096 bytes：

- 每个 page 包含一个 16 byte header；
- 每个 page 至少包含 2 条记录；
- 除了数据本身，每条记录还需要额外 8 bytes

$$\frac{4096-16}{2}=2040$$

因此当 (8+key+value) 组合大小不超过 2040 bytes 是最节省空间。

当存储的记录超过 page 大小，则将部分内容单独写入一个或多个专用 page。由于最后一个 page 余下部分不能与其它记录共享，因此当其大小为 (4096-16) bytes 的倍数时，更节省空间。

可以用 `Environment.stat()` 查看空间使用情况：

```python
>>> pprint(env.stat())
{'branch_pages': 1040L,
 'depth': 4L,
 'entries': 3761848L,
 'leaf_pages': 73658L,
 'overflow_pages': 0L,
 'psize': 4096L}
```

该数据库包含 3,761,848 条记录，没有溢出值（`overflow_pages`）。`Environment.stat` 仅返回默认数据库的信息。

记录的 key 长度默认限制为 511 bytes，不过可以通过重构库来调整。编译时 key 长度可以通过 `Environment.max_key_size()` 查询。

## 内存使用

与堆内存不同，文件支持的内存映射 page，如 LMDB 使用的 pages，只要 page 是干净的，就能随时被 OS 有效回收。*干净* 指缓存个的 page 内容与支持映射的 disk 中的相关 page 相匹配。实际上就是 [OS page 缓存](http://en.wikipedia.org/wiki/Page_cache)。

在 Linux 中，`/proc/<pid>/smaps` 文件包含进程中每个内存映射的一部分。要检查 LMDB 实际的内存使用情况，可以查找 `data.mdb` 项，检查其 `Dirty` 和 `Clean` 值。

当没有处于活动状态的 write 事务时，LMDB 数据库中所有 page 都应标记为 `clean`，除非 `Environment` 是以 `sync=False` 打开的，且上次 write 事务后没有显式调用 `Environment.sync()`，并且 OS writeback 机制还没有将 dirty 页面写入 disk。 

## Bytestrings

LMDB 使用 bytestring，对应 Python<=2.7 为 `str()` 类型和 Python>=3.0 的 `bytes()` 类型。

由于 Python 2.x 的设计，对只包含 ASCII 字符的 `str()` 实例，由于被隐式编码为 ASCII，LMDB 也接受这类 Unicode 实例。但是不应该依赖这类行为，对 Unicode 值强烈建议显式进行编码和解码。

## Buffers

由于 LMDB 是内存映射的，所以在 kernel、数据库 lib 或应用程序不复制 key 或 value 的情况下也能访问数据。为了利用该特性，可以为 `Environment.begin()` 或 `Transaction` 加入 `buffers=True` 设置，从而返回 `buffer()` 对象而不是 bytestring。

在 Python 中，`buffer()` 对象在许多地方可以替代 bytestrings。它们都像一个规则的序列：支持切片、索引、迭代或获取长度。许多 Python API 都会根据需要自动将 buffer 转换为 bytestring：

```python
>>> txn = env.begin(buffers=True)
>>> buf = txn.get('somekey')
>>> buf
<read-only buffer ptr 0x12e266010, size 4096 at 0x10d93b970>

>>> len(buf)
4096
>>> buf[0]
'a'
>>> buf[:2]
'ab'
>>> value = bytes(buf)
>>> len(value)
4096
>>> type(value)
<type 'bytes'>
```

也可以将 buffer 直接传递给许多本地 API，例如 `file.write()`, `socket.send()`, `zlib.decompress()` 等。将 buffer 传入另一个 buffer 可以不复制进行切片：

```python
>>> # Extract bytes 10 through 210:
>>> sub_buf = buffer(buf, 10, 200)
>>> len(sub_buf)
200
```

在 PyPy 和 CPython 中，返回的 buffer 在事务完成或以任何方式修改后会被丢弃。可以用 `bytes()` 复制缓冲区内容以保存：

```python
with env.begin(write=True, buffers=True) as txn:
    buf = txn.get('foo')           # only valid until the next write.
    buf_copy = bytes(buf)          # valid forever
    txn.delete('foo')              # this is a write!
    txn.put('foo2', 'bar2')        # this is also a write!

    print('foo: %r' % (buf,))      # ERROR! invalidated by write
    print('foo: %r' % (buf_copy,)) # OK

print('foo: %r' % (buf,))          # ERROR! also invalidated by txn end
print('foo: %r' % (buf_copy,))     # still OK
```

## writemap 模式

当以 `writemap=True` 设置调用 `Environment` 或 `open()`，lmdb 会以 writable 内存映射直接更新存储，以安全为代价提高性能，即在 Python 进程中可能存在有 bug 的 C 代码（尽管可能性不大）覆盖映射，导致数据库损坏。

> **WARNING:** 该选项可能导致不支持稀疏文件的文件系统（如 OSX）在首次打开或关闭环境时立即预分配 `map_size=` 字节的底层存储。

> **WARNING:** 如果启用该选项，文件系统故障（如空间不足）会使 Python 进程崩溃。当然这不属于 LMDB 的问题。

## 资源管理

`Environment`, `Transaction` 和 `Cursor` 都支持上下文管理器：

```python
with env.begin() as txn:
    with txn.cursor() as curs:
        # do stuff
        print 'key is:', curs.get('key')
```

在 CFFI 上使用 `Cursor` 上下文管理器很重要，如果在一个 transaction 中创建了多个 `Cursor`，则应该显式调用 `Cursor.close()`。在 CFFI 上不关闭 cursor 会导致积累许多无用对象，直到 parent transaction 中止或提交。

## 使用

使用流程：

- `env = lmdb.open()`：创建 lmdb 环境
- `txn = env.begin()`：建立事务
- `txn.put(key, value)`：进行插入和修改
- `txn.delete(key)`：进行删除
- `txn.get(key)`：进行查询
- `txn.cursor()`：进行遍历
- `txn.commit()`：提交更改

注意：

- `put` 和 `delete` 后一定要 `commit` ，不然根本没有存进去
- 每一次 `commit` 后，需要再定义一次 `txn=env.begin(write=True)`

### 创建 Environment

```python
# 安装：pip install lmdb
import lmdb

env = lmdb.open(lmdb_path, map_size=1099511627776)
```

`lmdb_path` 指定存放生成的lmdb数据库的文件夹路径，如果没有该文件夹则自动创建。

`map_size` 指定创建的新数据库所需磁盘空间的最小值，1099511627776B＝１T

会在指定路径下创建 `data.mdb` 和 `lock.mdb` 两个文件，一是个数据文件，一个是锁文件。

### 修改数据库内容

```python
txn = env.begin(write=True)

# 插入或修改
txn.put(str(1).encode(), "Alice".encode())
txn.put(str(2).encode(), "Bob".encode())

# 删除
txn.delete(str(1).encode())

txn.commit()
```

先创建一个事务（transaction）对象 `txn`，所有操作都经过这个事务对象。因为要对数据库进行写入操作，所以设置 `write=True`，其默认为 `False`。

使用 `.put(key, value)` 对数据库进行插入和修改操作，传入的参数为键值对。

另外，需要在字符串后加 `.encode()` 将 `str` 转换为 `bytes`，否则会报错：`TypeError: Won't implicitly convert Unicode to bytes; use .encode()`。后面再使用 `.decode()` 对其进行解码得到原数据。

使用 `.delete(key)` 删除指定键值对。

对 LMDB 的读写操作，最后需要使用 `commit` 提交待处理的事务。

### 查询数据库

```python
txn = env.begin()

print(txn.get(str(2).encode()))

for key, value in txn.cursor():
    print(key, value)

env.close()
```

每次 `commit()` 之后都要用 `env.begin()` 更新 txn，以得到最新的 lmdb 数据库。

使用 `.get(key)` 查询单条记录。

使用 `.cursor()` 遍历数据库中的所有记录，返回一个可迭代对象。

也可以使用 `with` 语法：

```python
with env.begin() as txn:
    print(txn.get(str(2).encode()))

    for key, value in txn.cursor():
        print(key, value)
```

### 完整演示

```python
import lmdb
import os, sys

def initialize():
    env = lmdb.open("lmdb_dir")
    return env

def insert(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def delete(env, sid):
    txn = env.begin(write=True)
    txn.delete(str(sid).encode())
    txn.commit()

def update(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def search(env, sid):
    txn = env.begin()
    name = txn.get(str(sid).encode())
    return name

def display(env):
    txn = env.begin()
    cur = txn.cursor()
    for key, value in cur:
        print(key, value)


env = initialize()

print("Insert 3 records.")
insert(env, 1, "Alice")
insert(env, 2, "Bob")
insert(env, 3, "Peter")
display(env)

print("Delete the record where sid = 1.")
delete(env, 1)
display(env)

print("Update the record where sid = 3.")
update(env, 3, "Mark")
display(env)

print("Get the name of student whose sid = 3.")
name = search(env, 3)
print(name)

# 最后需要关闭关闭lmdb数据库
env.close()

# 执行系统命令
os.system("rm -r lmdb_dir")
```

## API

### lmdb.open

```python
lmdb.open(path, **kwargs)
```

`Environment` 构造函数的快捷方式。

### lmdb.version

```python
lmdb.version(subpatch=False)
```

返回绑定的 LMDB 库版本，tuple 格式 `(major, minor, patch)`。Python 绑定的版本可以通过 `lmdb.__version__` 查看。

**参数：**

- `subpatch`

`True` 表示返回一个 4 整数 tuple，前面 3 个同上，最后一个整数表示由 py-lmdb 应该的补丁（0 表示没有）。

### Environment

```python
class lmdb.Environment(
    path, 
    map_size=10485760, 
    subdir=True, 
    readonly=False, 
    metasync=True, 
    sync=True, 
    map_async=False, 
    mode=493, 
    create=True, 
    readahead=True, 
    writemap=False, 
    meminit=True, 
    max_readers=126, 
    max_dbs=0, 
    max_spare_txns=1, 
    lock=True)
```

数据库环境。一个环境可以包含多个数据库，位于同一个内存映射或底层磁盘文件中。

写操作要先创建 `Transaction`。一次只能有一个 write `Transaction`，read `Transaction` 数不限制（即使已有一个 write `Transaction`）。

该类的别名为 `lmdb.open`。等价于 C 库的 [mdb_env_open()](http://www.lmdb.tech/doc/group__mdb.html)。

在同一进程同时打开相同的 LMDB 文件会导致严重错误，可能导致数据损坏和解释器崩溃。

**参数：**

- `path`

存储数据库的目录（`subdir=True`）或存储的数据库文件前缀。

- `map_size=10485760`

设置此环境的内存映射大小，内存映射大小也是数据的最大大小，单位 bytes。如果数据库增加到大于 `map_size`，会抛出异常，必须关闭并重新打开新的 `Environment`。在 64 位系统上，创建一个非常大的空间（如 1TB）没问题，在 32 位系统上则必须小于 2GB。

> 默认的 `map_size` 很小，很容易抛出异常，以方便用户理解该选项，从而选择一个合适的值。

`map_size` 应该是操作系统页面大小的倍数，且应该尽可能大，以使用数据库可能的增长。

- `subdir=True`

`True` 时 `path` 指的是存储数据和锁文件的子目录，否则指的是文件名前缀。

- `readonly=False`

`True` 时不允许任何写操作。不过 lock 文件仍然被修改。指定后忽略传入 `begin()` 或 `Transaction` 的 `write` 选项。

- `readahead=True`

`False` 时 LMDB 禁用 OS 文件系统的读取机制，在数据库大于 RAM 时可能提高随机读取性能。

- `meminit=True`

`False` 表示 LMDB 在将缓冲区写入 disk 前不会执行0 初始化。这会提高性能，但会导致 old heap 数据保存在缓冲区的未使用部分。对重要数据（如明文密码）不要使用。此选项只有在 `writemap=False` 才有意义，在 `writemap=True` 时新的页面必然执行 0 初始化。

- `max_readers=126`

同时进行的读 transaction 最大数量。只能由打开环境的第一个进程设置，因为它会影响锁文件和共享内存区的大小。

同时启用的的 *read* transaction 数大于该值会失败。

- `lock=True`

`False` 表示不锁定。如果要并发，由调用者管理所有并发。为了保证操作正确，调用方必须保证只有一个 *write* 语义，且在写入状态时没有 *reader* 使用旧的 transaction。最简单的方法是使用独占锁，以便在写入时根本无法激活 *reader*。

**方法：**

```python
begin(db=None, parent=None, write=False, buffers=False)
```

`lmdb.Transaction` 的快捷方式。

- `stat()`

以 dict 返回环境默认数据库的统计信息：

|统计项|说明|
|---|---|
|psize|数据库页面大小（byte）|
|depth|B-tree 高度|
|branch_pages|Number of internal (non-leaf) pages.|
|leaf_pages|Number of leaf pages.|
|overflow_pages|Number of overflow pages.|
|entries|数据个数|

### Transaction

```python
class lmdb.Transaction(env, 
    db=None, 
    parent=None, 
    write=False, 
    buffers=False)
```

transaction 对象。所有操作都需要一个 transaction 句柄，transaction 可以是只读，也可以是读写。写 transaction 不能跨线程。transaction 对象实现了上下文管理器协议，因此即使遇到异常，transaction 也能可靠释放：

```python
# Transaction aborts correctly:
with env.begin(write=True) as txn:
    crash()

# Transaction 自动 commits
with env.begin(write=True) as txn:
    txn.put('a', 'b')
```

等价于 `mdb_txn_begin()`。

**参数：**

- `env`

事务所处的环境。

- `db`

要操作的默认命名数据库。如果未指定，默认为环境的主数据库。

- `parent`

`None` 或者 父 事务。



**方法：**

- `get(key, default=None, db=None)`

返回第一个与 `key` 匹配的值，如果没找到 `key`，返回 `default`。对 `dupsort=True` 数据库，必须使用 cursor 来获得指定 key 的所有值。

等价于 `mdb_get()`。

- `stat(db)`

### Cursor

```python
class lmdb.Cursor(db, txn)
```

用于访问数据库，等价于 C 库的 [mdb_cursor_open()](http://www.lmdb.tech/doc/group__mdb.html#ga9ff5d7bd42557fd5ee235dc1d62613aa)。

**参数：**

- `db`

要访问的 `_Database`。

- `txn`

要访问的 `Transaction`。

使用 `Transaction.cursor()` 快速获得 cursor：

```python
>>> env = lmdb.open('/tmp/foo')
>>> child_db = env.open_db('child_db')
>>> with env.begin() as txn:
...     cursor = txn.cursor()           # Cursor on main database.
...     cursor2 = txn.cursor(child_db)  # Cursor on child database.
```

Cursor 开始时处于未定位状态。调用 `iternext()` 或 `iterprev()` 

## 参考

- https://lmdb.readthedocs.io/en/release/
