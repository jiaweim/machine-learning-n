# CIFAR-10

## 下载数据集

使用 `torchvision` 下载数据集：

```python
from torchvision import datasets

data_path = 'data/cifar10'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)
```

