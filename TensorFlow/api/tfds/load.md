# tfds.load

## 简介

加载指定名称的数据集，返回 `tf.data.Dataset`：

```python
tfds.load(
    name: str,
    *,
    split: Optional[Tree[splits_lib.Split]] = None,
    data_dir: Optional[str] = None,
    batch_size: tfds.typing.Dim = None,
    shuffle_files: bool = False,
    download: bool = True,
    as_supervised: bool = False,
    decoders: Optional[TreeDict[decode.Decoder]] = None,
    read_config: Optional[tfds.ReadConfig] = None,
    with_info: bool = False,
    builder_kwargs: Optional[Dict[str, Any]] = None,
    download_and_prepare_kwargs: Optional[Dict[str, Any]] = None,
    as_dataset_kwargs: Optional[Dict[str, Any]] = None,
    try_gcs: bool = False
)
```

`tfds.load` 可用于执行如下操作：

1. 通过名称获取 `tfds.core.DatasetBuilder`

```python
builder = tfds.builder(name, data_dir=data_dir, **builder_kwargs)
```

## 参数

- `batch_size`

int, 给数据集添加 batch 维度。可变长度的 features 以 0 填充。如果 `batch_size=-1`，则整个数据集为 1 个 batch。

## 参考

- https://www.tensorflow.org/datasets/api_docs/python/tfds/load
