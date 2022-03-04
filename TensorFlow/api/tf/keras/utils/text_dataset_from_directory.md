# tf.keras.utils.text_dataset_from_directory

2022-03-04, 00:44
***

## 简介

```python
tf.keras.utils.text_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, batch_size=32, max_length=None, shuffle=True, seed=None,
    validation_split=None, subset=None, follow_links=False
)
```

使用目录中的文本文件生成 [tf.data.Dataset](../../data/Dataset.md)。

如果你的目录结构为：

```sh
main_directory/
...class_a/
......a_text_1.txt
......a_text_2.txt
...class_b/
......b_text_1.txt
......b_text_2.txt
```

则调用 `text_dataset_from_directory(main_directory, labels='inferred')` 

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
