# HDF5 并行

- [HDF5 并行](#hdf5-并行)
  - [简介](#简介)
  - [HDF5 并行原理](#hdf5-并行原理)
  - [构建并行 HDF5](#构建并行-hdf5)
  - [h5py 中使用并行 HDF5](#h5py-中使用并行-hdf5)
  - [集体操作 vs. 独立操作](#集体操作-vs-独立操作)
  - [MPI 原子模式](#mpi-原子模式)
  - [参考](#参考)

Last updated: 2023-02-06, 15:18
****

## 简介

HDF5 文件的只读并行访问不需要特别处理，每个进程应该单独打开文件进行读取。

[HDF5 并行](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) 是一个建立在 MPI 上的特性，用于支持并行写入 HDF5 文件。要使用该特性，HDF5 和 h5py 都必须在编译时开启 MPI。

## HDF5 并行原理

HDF5 并行是 HDF5 库的一个配置选项，用于支持在多个并行进程间共享打开的文件。它使用 MPI（Message Passing Interface）标准实现进程间通信。因此，在 Python 中使用并行 HDF5 ，必须同时使用 MPI 库。`mpi4py` 提供了 MPI 的 Python 绑定。使用 `mpi4py` 实现 "Hello World"：

```python
from mpi4py import MPI
print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())
```

对基于 mpi 的并行程序，使用 `mpiexec` 程序启动并行 Python 实例：

```powershell
$ mpiexec -n 4 python demo.py
Hello World (from process 1)
Hello World (from process 2)
Hello World (from process 3)
Hello World (from process 0)
```

`mpi4py` 包含进程间共享数据、同步等各种机制，与线程或 `multiprocessing` 的并行原理不同。

详细信息可参考 [mpi4py 主页](https://github.com/mpi4py/mpi4py/)。

## 构建并行 HDF5

HDF5 必须至少使用以下选项构建：

```powershell
$./configure --enable-parallel --enable-shared
```

通常可以从包管理器获得 HDF5 的并行版本。可以使用 `h5cc` 程序查看构建选项；

```powershell
$ h5cc -showconfig
```

获得 HDF5 的并行构建后，h5py 还需要以 "MPI 模式" 进行编译。姜末热恩编译器设置为 "mpicc" wrapper，并使用 `HDF5_MPI` 环境构建 h5py：

```powershell
$ export CC=mpicc
$ export HDF5_MPI="ON"
$ export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
$ pip install .
```

## h5py 中使用并行 HDF5

HDF5 的并行特性基本是透明的。使用 `mpio` 文件驱动打开多进程共享文件。下面示例打开一个文件，创建一个数据集，并将进程的 ID 写入数据集：

```python
from mpi4py import MPI
import h5py

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)

dset = f.create_dataset('test', (4,), dtype='i')
dset[rank] = rank

f.close()
```

运行程序：

```powershell
$ mpiexec -n 4 python demo2.py
```

使用 `h5dump` 查看文件：

```powershell
$ h5dump parallel_test.hdf5
HDF5 "parallel_test.hdf5" {
GROUP "/" {
   DATASET "test" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
      DATA {
      (0): 0, 1, 2, 3
      }
   }
}
}
```

## 集体操作 vs. 独立操作

基于 MPI 的程序通过启动多个 Python 解释器来工作，每个解释器实例都运行脚本。对每个进程可以做什么，都有一定要求。HDF5 的某些操作，如修改文件元数据，必须所有进程都能执行，而其它操作，如写入数据到数据集，可以由某些进程执行，而其它进程不可以。

这两类操作被称为集体操作和独立操作。任何修改文件结构或元数据的操作都必须是集体操作，例如，在创建 group 时，每个进程都必须参与：

```python
>>> grp = f.create_group('x')  # right
>>> if rank == 1:
...     grp = f.create_group('x')   # wrong; all processes must do this
```

而将数据写入数据集，可以独立完成：

```python
if rank > 2:
    dset[rank] = 42   # this is fine
```

## MPI 原子模式

HDF5 1.8.9+  支持 MPI 原子文件访问模式，牺牲性能换取更严格的一致性要求。一旦用 `mpio` 驱动打开文件，可以使用 `atomic` 属性启用原子模式：

```python
f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
f.atomic = True
```

## 参考

- https://docs.h5py.org/en/stable/mpi.html
