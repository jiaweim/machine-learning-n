# get_file

2022-03-01, 16:41
****

## 简介

```python
tf.keras.utils.get_file(
    fname=None,
    origin=None,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir='datasets',
    hash_algorithm='auto',
    extract=False,
    archive_format='auto',
    cache_dir=None
)
```

如果文件不在缓存中，则从指定 URL 下载该文件。**返回文件保存路径**。

默认从 URL `origin` 下载文件到缓存目录 `~/.keras` 下的子目录 `datasets`，文件命名为 `fname`。例如，下载文件 `example.txt` 的最终位置为 `~/.keras/datasets/example.txt`。

还可以解压 tar、tar.gz、tar.bz 以及 zip 格式的压缩文件。传入 hash 值可以在下载后验证文件。命令行程序 `shasum` 和 `sha256sum` 可用于计算哈希值。

## 参数

|参数|说明|
|---|---|
|fname|文件名。如果是绝对路径，如 `/path/to/file.txt`，则文件保存到该位置。如果为 `None`，则使用 `origin` 处文件名称|
|origin|文件的 URL|
|untar|*不推荐*，boolean, 是否解压缩文件。建议使用 `extract` 参数|
|md5_hash|*不推荐*，建议使用 `file_hash` 参数。用于验证文件的 md5 hash 值|
|file_hash|下载后文件的期望 hash 字符串。支持 sha256 和 md5 hash|
|cache_subdir|Keras 缓存目录下的保存文件的子目录。如果 `fname` 是绝对路径 `/path/to/folder`，则直接将文件保存到该位置|
|hash_algorithm|选择验证文件的 hash 算法。支持 'md5', 'sha256' 和 'auto'。默认 'auto' 自动检测使用的 hash 算法|
|extract|True 表示解压文件|
|archive_format|压缩文件格式。支持 'auto', 'tar', 'zip', and None 选项。'tar' 包括 tar, tar.gz 和 tar.bz. 默认 'auto' 对应 ['tar', 'zip']. None 或空 list 返回无匹配项|
|cache_dir|缓存目录，None 表示默认位置 `~/.keras/.`|

## 示例

```python
path_to_downloaded_file = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar=True)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
