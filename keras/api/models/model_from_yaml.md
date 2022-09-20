# tf.keras.models.model_from_yaml

Last updated: 2022-09-19, 16:28
****

## 简介

```python
tf.keras.models.model_from_yaml(
    yaml_string, custom_objects=None
)
```

解析 YAML 模型配置文件，返回模型实例（uncompiled）。

> **[!NOTE]** 从 TF 2.6 开始不再支持该方法，调用抛出 RuntimeError。

|参数|说明|
|---|---|
|yaml_string|YAML string 或编码模型配置的文件|
|custom_objects|(可选）将名称映射到反序列要使用的自定义类或函数|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_yaml
