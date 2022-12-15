# ImageFolder

## 简介

```python
torchvision.datasets.ImageFolder(root: str, transform: ~typing.Optional[~typing.Callable] = None, target_transform: ~typing.Optional[~typing.Callable] = None, loader: ~typing.Callable[[str], ~typing.Any] = <function default_loader>, is_valid_file: ~typing.Optional[~typing.Callable[[str], bool]] = None)
```

一种通用图像加载器，图像默认以如下方式存储：

```python
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

该类继承自 DatasetFolder，因此可以覆盖相同的方法来自定义数据集。

参数：

- **root** (`string`)：根目录
- **transform** (`callable`, optional)：接受 PIL 图像，返回变换版本图像的函数或变换，如 `transforms.RandomCrop`
- **target_transform** (`callable`, optional)：target 变换函数
- **loader** (`callable`, optional)：