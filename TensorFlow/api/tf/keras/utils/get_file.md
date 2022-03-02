# tf.keras.utils.get_file

2022-03-01, 16:41
***

## 简介

```python
tf.keras.utils.get_file(
    fname=None, origin=None, untar=False, md5_hash=None, file_hash=None,
    cache_subdir='datasets', hash_algorithm='auto',
    extract=False, archive_format='auto', cache_dir=None
)
```

如果文件不在缓存中，则从指定 URL 下载该文件。返回下载文件的路径。

默认从 URL `origin` 下载文件到缓存目录 `~/.keras` 下的子目录 `datasets`，文件命名为 `fname`。因此，文件 `example.txt` 的最终位置为 `~/.keras/datasets/example.txt`。

还可以提取tar、tar.gz、tar.bz 以及 zip 格式的压缩文件。传入 hash 值可以在下载后验证文件。命令行程序 `shasum` 和 `sha256sum` 可用于计算哈希值。

## 示例

```python
path_to_downloaded_file = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True)
```

## 参数

|参数|说明|
|---|---|
|fname|文件名称。如果是绝对路径，如 `/path/to/file.txt`，则文件会保存到该位置。如果为 `None`，则使用 `origin` 处文件名称|
|origin|文件的 URL|
|untar|Deprecated，建议使用 `extract` 参数。boolean, 是否解压缩文件|
|md5_hash|Deprecated，建议使用 `file_hash` 参数。文件的 md5 hash 值|
|file_hash|下载后文件的预期 hash 字符串。支持 sha256 和 md5 hash 算法|
|cache_subdir|保存文件的 Keras 缓存目录下的子目录。如果指定的是绝对路径 `/path/to/folder`，则文件将保存到该位置|
|hash_algorithm|选择验证文件的 hash 算法。选项包括 'md5', 'sha256' 和 'auto'。默认 'auto' 会检测正在使用的 hash 算法|
|extract|True tries extracting the file as an Archive, like tar or zip.|
|archive_format|Archive format to try for extracting the file. Options are 'auto', 'tar', 'zip', and None. 'tar' includes tar, tar.gz, and tar.bz files. The default 'auto' corresponds to ['tar', 'zip']. None or an empty list will return no matches found.|
|cache_dir|Location to store cached files, when None it defaults to the default directory ~/.keras/.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
