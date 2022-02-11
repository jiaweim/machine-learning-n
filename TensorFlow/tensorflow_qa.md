# TensorFlow QAs

## 证书问题

在下载 tensorflow.keras 数据集时遇到证书问题下载失败，可以添加如下语句解决该问题：

```python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```
