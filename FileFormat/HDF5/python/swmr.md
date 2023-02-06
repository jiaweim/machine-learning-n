# SWMR

- [SWMR](#swmr)
  - [简介](#简介)
  - [h5py 中使用 SWMR](#h5py-中使用-swmr)
  - [示例](#示例)
    - [使用 notify 监视 Dataset](#使用-notify-监视-dataset)
    - [多进程读写](#多进程读写)
  - [参考](#参考)

Last updated: 2023-02-06, 16:26
****

## 简介

SWMR（Single Writer Multiple Reader）允许在一个进程写入 HDF5 的同时，使用多个进程读取。在添加 SWMR 功能之前，由于数据和元数据不同步，无法读取正在写入的 HDF5 文件。

使用 SWMR 模式输出文件时可以保证文件总是可读。该模式的优点是，即时写入进程在正确关闭文件之前崩溃，文件也始终处于有效状态。

SWMR 的实现独立于写入或读取进程，进程之间不需要同步，由用户实现文件轮训机制、nontify 或其它 IPC 机制来通知何时写入数据。

SWMR 特性需要最新的 HDF5 格式：v110，因此要求 HDF5 版本不低于 1.10（可以使用 `h5py.version.info` 查看），并在打开或创建文件时将 `libver` 设置为 “latest”。

> **WARNING**
> v110 格式与 v18 格式不兼容，因此使用 SWMR 模式和 libver=’latest’ 写入的文件不能在老版本的 HDF5 库打开。

## h5py 中使用 SWMR

写入和读取进程通常需要按如下步骤进行：

- Writer 进程创建目标文件和所有 group, datasets 和 attributes；
- Writer 进程切换到 SWMR 模式；
- Reader 进程可以使用 `swmr=True` 打开文件；
- Writer 向现有 dataset 写入或追加数据（SWMR 模式下不能创建 dataset 和 group）；
- Writer 定期 flush 目标 dataset，使其对 reader 进程可见；
- Reader 在读取新的数据或元数据前刷新 target dataset；
- Writer 完成后正常关闭文件；
- Reader 可以根据需要关闭文件。

下面演示 SWMR writer 追加数据到单个 dataset：

```python
f = h5py.File("swmr.h5", 'w', libver='latest')
arr = np.array([1,2,3,4])
dset = f.create_dataset("data", chunks=(2,), maxshape=(None,), data=arr)
f.swmr_mode = True # 切换到 SWMR 模式
# Now it is safe for the reader to open the swmr.h5 file
for i in range(5):
    new_shape = ((i+1) * len(arr), )
    dset.resize( new_shape )
    dset[i*len(arr):] = arr
    dset.flush()
    # Notify the reader process that new data has been written
```

下面演示如何在 SWMR Reader 中监视数据：

```python
f = h5py.File("swmr.h5", 'r', libver='latest', swmr=True)
dset = f["data"]
while True:
    dset.id.refresh()
    shape = dset.shape
    print( shape )
```

## 示例

### 使用 notify 监视 Dataset

本例使用 linux inotify (Python 绑定为 `pyinotify`)实现在目标文件更新时接受一个信号：

```python
"""
    Demonstrate the use of h5py in SWMR mode to monitor the growth of a dataset
    on notification of file modifications.

    This demo uses pyinotify as a wrapper of Linux inotify.
    https://pypi.python.org/pypi/pyinotify

    Usage:
            swmr_inotify_example.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmr.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will open the file in SWMR mode and monitor the shape of the
    dataset on every write event (from inotify). If another application is
    concurrently writing data to the file, the writer must have have switched
    the file into SWMR mode before this script can open the file.
"""
import asyncore
import pyinotify
import sys
import h5py
import logging

#assert h5py.version.hdf5_version_tuple >= (1,9,178), "SWMR requires HDF5 version >= 1.9.178"

class EventHandler(pyinotify.ProcessEvent):

    def monitor_dataset(self, filename, datasetname):
        logging.info("Opening file %s", filename)
        self.f = h5py.File(filename, 'r', libver='latest', swmr=True)
        logging.debug("Looking up dataset %s"%datasetname)
        self.dset = self.f[datasetname]

        self.get_dset_shape()

    def get_dset_shape(self):
        logging.debug("Refreshing dataset")
        self.dset.refresh()

        logging.debug("Getting shape")
        shape = self.dset.shape
        logging.info("Read data shape: %s"%str(shape))
        return shape

    def read_dataset(self, latest):
        logging.info("Reading out dataset [%d]"%latest)
        self.dset[latest:]

    def process_IN_MODIFY(self, event):
        logging.debug("File modified!")
        shape = self.get_dset_shape()
        self.read_dataset(shape[0])

    def process_IN_CLOSE_WRITE(self, event):
        logging.info("File writer closed file")
        self.get_dset_shape()
        logging.debug("Good bye!")
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s  %(levelname)s\t%(message)s',level=logging.INFO)

    file_name = "swmr.h5"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    dataset_name = "data"
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]


    wm = pyinotify.WatchManager()  # Watch Manager
    mask = pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_WRITE
    evh = EventHandler()
    evh.monitor_dataset( file_name, dataset_name )

    notifier = pyinotify.AsyncNotifier(wm, evh)
    wdd = wm.add_watch(file_name, mask, rec=False)

    # Sit in this loop() until the file writer closes the file
    # or the user hits ctrl-c
    asyncore.loop()
```

### 多进程读写

SWMR 多进程示例启动两个子进程：reader 和 writer 各一个。writer 进程首先创建目标文件和 dataset。然后切换到 SWMR 模式，通知 reader 进程（通过 `multiprocessing.Event`）可以安全打开文件进行读操作。

writer 进程继续向 dataset 追加数据。每次写入后，它都会通知 reader 写入了新数据。此时新数据是否在文件中可见取决于 OS 和文件系统延迟。

reader 先等待来自 writer 的 SWMR 模式通知，然后进入循环，在循环中等待来自 writer 的下一步通知。每次接到通知，reader 都会刷新 dataset 并读取维度。超时后，reader 会退出循环。

```python
"""
    Demonstrate the use of h5py in SWMR mode to write to a dataset (appending)
    from one process while monitoring the growing dataset from another process.

    Usage:
            swmr_multiprocess.py [FILENAME [DATASETNAME]]

              FILENAME:    name of file to monitor. Default: swmrmp.h5
              DATASETNAME: name of dataset to monitor in DATAFILE. Default: data

    This script will start up two processes: a writer and a reader. The writer
    will open/create the file (FILENAME) in SWMR mode, create a dataset and start
    appending data to it. After each append the dataset is flushed and an event
    sent to the reader process. Meanwhile the reader process will wait for events
    from the writer and when triggered it will refresh the dataset and read the
    current shape of it.
"""

import sys
import h5py
import numpy as np
import logging
from multiprocessing import Process, Event

class SwmrReader(Process):
    def __init__(self, event, fname, dsetname, timeout = 2.0):
        super().__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname
        self._timeout = timeout

    def run(self):
        self.log = logging.getLogger('reader')
        self.log.info("Waiting for initial event")
        assert self._event.wait( self._timeout )
        self._event.clear()

        self.log.info("Opening file %s", self._fname)
        f = h5py.File(self._fname, 'r', libver='latest', swmr=True)
        assert f.swmr_mode
        dset = f[self._dsetname]
        try:
            # monitor and read loop
            while self._event.wait( self._timeout ):
                self._event.clear()
                self.log.debug("Refreshing dataset")
                dset.refresh()

                shape = dset.shape
                self.log.info("Read dset shape: %s"%str(shape))
        finally:
            f.close()

class SwmrWriter(Process):
    def __init__(self, event, fname, dsetname):
        super().__init__()
        self._event = event
        self._fname = fname
        self._dsetname = dsetname

    def run(self):
        self.log = logging.getLogger('writer')
        self.log.info("Creating file %s", self._fname)
        f = h5py.File(self._fname, 'w', libver='latest')
        try:
            arr = np.array([1,2,3,4])
            dset = f.create_dataset(self._dsetname, chunks=(2,), maxshape=(None,), data=arr)
            assert not f.swmr_mode

            self.log.info("SWMR mode")
            f.swmr_mode = True
            assert f.swmr_mode
            self.log.debug("Sending initial event")
            self._event.set()

            # Write loop
            for i in range(5):
                new_shape = ((i+1) * len(arr), )
                self.log.info("Resizing dset shape: %s"%str(new_shape))
                dset.resize( new_shape )
                self.log.debug("Writing data")
                dset[i*len(arr):] = arr
                #dset.write_direct( arr, np.s_[:], np.s_[i*len(arr):] )
                self.log.debug("Flushing data")
                dset.flush()
                self.log.info("Sending event")
                self._event.set()
        finally:
            f.close()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)10s  %(asctime)s  %(name)10s  %(message)s',level=logging.INFO)
    fname = 'swmrmp.h5'
    dsetname = 'data'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    if len(sys.argv) > 2:
        dsetname = sys.argv[2]

    event = Event()
    reader = SwmrReader(event, fname, dsetname)
    writer = SwmrWriter(event, fname, dsetname)

    logging.info("Starting reader")
    reader.start()
    logging.info("Starting reader")
    writer.start()

    logging.info("Waiting for writer to finish")
    writer.join()
    logging.info("Waiting for reader to finish")
    reader.join()
```

下面的输出示例演示了 reader 和 writer 之间的延迟：

```powershell
python examples/swmr_multiprocess.py
  INFO  2015-02-26 18:05:03,195        root  Starting reader
  INFO  2015-02-26 18:05:03,196        root  Starting reader
  INFO  2015-02-26 18:05:03,197      reader  Waiting for initial event
  INFO  2015-02-26 18:05:03,197        root  Waiting for writer to finish
  INFO  2015-02-26 18:05:03,198      writer  Creating file swmrmp.h5
  INFO  2015-02-26 18:05:03,203      writer  SWMR mode
  INFO  2015-02-26 18:05:03,205      reader  Opening file swmrmp.h5
  INFO  2015-02-26 18:05:03,210      writer  Resizing dset shape: (4,)
  INFO  2015-02-26 18:05:03,212      writer  Sending event
  INFO  2015-02-26 18:05:03,213      reader  Read dset shape: (4,)
  INFO  2015-02-26 18:05:03,214      writer  Resizing dset shape: (8,)
  INFO  2015-02-26 18:05:03,214      writer  Sending event
  INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (12,)
  INFO  2015-02-26 18:05:03,215      writer  Sending event
  INFO  2015-02-26 18:05:03,215      writer  Resizing dset shape: (16,)
  INFO  2015-02-26 18:05:03,215      reader  Read dset shape: (12,)
  INFO  2015-02-26 18:05:03,216      writer  Sending event
  INFO  2015-02-26 18:05:03,216      writer  Resizing dset shape: (20,)
  INFO  2015-02-26 18:05:03,216      reader  Read dset shape: (16,)
  INFO  2015-02-26 18:05:03,217      writer  Sending event
  INFO  2015-02-26 18:05:03,217      reader  Read dset shape: (20,)
  INFO  2015-02-26 18:05:03,218      reader  Read dset shape: (20,)
  INFO  2015-02-26 18:05:03,219        root  Waiting for reader to finish
```

## 参考

- https://docs.h5py.org/en/stable/swmr.html
