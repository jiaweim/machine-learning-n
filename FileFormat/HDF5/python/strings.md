# HDF5 中字符串读写

## 保存字符串

当创建新的 dataset 或 attribute，Python `str` 和 `bytes` 对象被视为变长字符串，分别标记为 UTF-8 和 ASCII。NumPy 字节数组 `S` 为定长字符串。