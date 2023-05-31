# DatasetFolder

****

## 简介

```python
torchvision.datasets.DatasetFolder(root: str, loader: Callable[[str], Any], extensions: Optional[Tuple[str, ...]] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None)
```

通用数据加载器。

默认目录结构可以通过覆盖 `find_classes()` 方法自定义。

## 参数

- **root** (`string`)：根目录
-  **loader** (`callable`)：根据路径加载样本的函数
-  **extensions** (`tuple[string]`)：允许的扩展名列表，不要同时设置 `extensions` 和 `is_valid_file`
-  **transform** (`callable`, optional) ：输入变换函数，如 `transforms.RandomCrop`
-  **target_transform** (`callable`, optional)：target 变换函数
-  **is_valid_file** ：根据文件路径，检查该文件是否为有效文件

## 方法

```python
find_classes(directory: str) → Tuple[List[str], Dict[str, int]]
```

按如下数据集结构查找不同类别文件：

```txt
directory/
├── class_x
│   ├── xxx.ext
│   ├── xxy.ext
│   └── ...
│       └── xxz.ext
└── class_y
    ├── 123.ext
    ├── nsdf3.ext
    └── ...
    └── asd932_.ext
```

可以重写该方法，以使用不同的数据集目录结构。

参数：

- **directory** (`str`)：根目录，对应 `self.root`

返回：所有类别的列表，以及类别到索引的 dict。

```python
static make_dataset(directory: str, class_to_idx: Dict[str, int], extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[str], bool]] = None) → List[Tuple[str, int]]
```

## 参考

- https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html
