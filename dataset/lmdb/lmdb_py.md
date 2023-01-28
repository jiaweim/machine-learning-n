# lmdb

- [lmdb](#lmdb)
  - [简介](#简介)
  - [安装：Windows](#安装windows)
  - [使用](#使用)
    - [创建 lmdb 环境](#创建-lmdb-环境)
    - [修改数据库内容](#修改数据库内容)
    - [查询数据库](#查询数据库)
    - [完整演示](#完整演示)
  - [API](#api)
    - [lmdb.open](#lmdbopen)
    - [Environment](#environment)
    - [Transaction](#transaction)
  - [参考](#参考)

***

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

## 使用

- `env = lmdb.open()`：创建 lmdb 环境
- `txn = env.begin()`：建立事务
- `txn.put(key, value)`：进行插入和修改
- `txn.delete(key)`：进行删除
- `txn.get(key)`：进行查询
- `txn.cursor()`：进行遍历
- `txn.commit()`：提交更改

### 创建 lmdb 环境

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
    create=True, readahead=True, writemap=False, meminit=True, max_readers=126, max_dbs=0, max_spare_txns=1, lock=True)
```

数据库环境。一个环境可以包含多个数据库，它们都位于同一个内存映射或底层磁盘文件中。

要写入环境必须先创建 `Transaction`。一次只能进行一个写 `Transaction`，不限制读 `Transaction` 数目，即使已有一个写 `Transaction`。

该类的别名为 `lmdb.open`，同时等价于 `mdb_env_open()`。

在同一进程同时打开相同的 LMDB 文件会导致严重错误。如果不注意这一点，可能会导致数据损坏和解释器崩溃。

**参数：**

- `path`

存储数据库的目录（`subdir=True` 时）或存储数据库的文件前缀。

- `map_size=10485760`

数据库可以增长到的最大大小，用于调整内存映射大小。如果数据库增加到大于 `map_size`，会抛出异常，必须关闭并重新打开新的 `Environment`。在 64 位系统上，创建一个非常大的空间（如 1TB）没问题，在 32 位系统上则必须小于 2GB。

> 默认的 `map_size` 很小，很容易抛出异常，以方便用户理解该选项，从而选择一个合适的值。

- `subdir=True`

`True` 时 `path` 指的是存储数据和锁文件的子目录，否则指的是文件名前缀。

- `readonly=False`

`True` 时不允许任何写操作。不过 lock 文件仍然被修改。指定后忽略传入 `begin()` 或 `Transaction` 的 `write` 选项。

**方法：**

```python
begin(db=None, parent=None, write=False, buffers=False)
```

`lmdb.Transaction` 的快捷方式。

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



## 参考

- https://lmdb.readthedocs.io/en/release/
