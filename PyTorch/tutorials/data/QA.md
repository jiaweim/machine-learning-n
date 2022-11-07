# PyTorch Data QAs

## 下载数据集失败

以 FashionMNIST 数据集为例：

- 先从 github 下载数据集：https://github.com/zalandoresearch/fashion-mnist

![](images/2022-11-07-11-05-58.png)

这 4 个文件都需要下载下来。

- 在代码中的 `root` 目录，创建 FashionMNIST 目录：

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

将下载好的数据放入该目录的 `raw` 文件夹。
