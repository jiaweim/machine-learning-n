# 快速入门

2026-09-10: 基于 2.14.0 API 修改
@since 2023-02-06⭐
@author Jiawei Mao

***

## 简介

下面介绍机器学习中常见任务的相关 API。

## 数据处理

PyTorch 提供两个处理数据的基础类：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。`Dataset` 包含数据样本及其标签，`DataLoader` 将 `Dataset` 封装为可迭代对象。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
```

PyTorch 提供了多个特定领域的扩展库，如 [TorchText](https://pytorch.org/text/stable/index.html), [TorchVision](https://pytorch.org/vision/stable/index.html) 和 [TorchAudio](https://pytorch.org/audio/stable/index.html)，这些库都内置了相关数据集。本教程将使用 TorchVision 提供的一个数据集。

`torchvision.datasets` 模块包含许多真实视觉数据的 `Dataset` 对象，如 CIFAR，COCO [等](https://pytorch.org/vision/stable/datasets.html)，下面使用 FashionMNIST 数据集。每个 TorchVision `Dataset` 包含两个参数：`transform` 和 `target_transform`，分别用于对样本（图像）和标签进行转换/修改。

```python
# 下载训练集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
# 下载测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
```

将 `Dataset` 作为参数传递给 `DataLoader`，它在数据集上封装了一个迭代器，并支持自动批处理（batching）、采样（sampling）、数据打乱（shuffling）以及多进程数据加载。下面将 batch size 设置为 64，即 dataloader 迭代器的每个元素包含 64 对样本和标签。

```python
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

```txt
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

## 创建模型

在 PyTorch 中通过继承 `nn.Module` 类定义神经网络：

- 在 `__init__` 函数中定义网络层
- 在 `forward` 函数中指定数据在网络中的前向传播。

为了加速神经网络计算，可以将模型移到硬件加速器，如 CUDA, MPS, MTIA 或 XPU 等。

```python
# 优先使用 GPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

```txt
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

## 优化模型参数

训练模型，我们需要一个[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)和一个[优化器](https://pytorch.org/docs/stable/optim.html)：

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在单个训练循环中，模型对训练数据集（按 batch 输入）进行预测，并根据反向传播预测误差来调整模型的参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

在测试集上评估模型性能。

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

训练过程经过多次迭代（epochs）。在每个 epoch，模型会不断调整参数以做出更好的预测。我们在每个 epoch 结尾打印模型的准确率和损失值，我们期望随着 epoch 增加，准确率不断提高，而损失值则不断下降。

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

```txt
Epoch 1
-------------------------------
loss: 2.310277  [   64/60000]
loss: 2.291593  [ 6464/60000]
loss: 2.267050  [12864/60000]
loss: 2.259090  [19264/60000]
loss: 2.241936  [25664/60000]
loss: 2.211771  [32064/60000]
loss: 2.224165  [38464/60000]
loss: 2.185110  [44864/60000]
loss: 2.190119  [51264/60000]
loss: 2.158839  [57664/60000]
Test Error: 
 Accuracy: 52.0%, Avg loss: 2.145115 

Epoch 2
-------------------------------
loss: 2.155910  [   64/60000]
loss: 2.143800  [ 6464/60000]
loss: 2.080454  [12864/60000]
loss: 2.104686  [19264/60000]
loss: 2.051725  [25664/60000]
loss: 1.985721  [32064/60000]
loss: 2.032461  [38464/60000]
loss: 1.938378  [44864/60000]
loss: 1.959611  [51264/60000]
loss: 1.893426  [57664/60000]
Test Error: 
 Accuracy: 53.5%, Avg loss: 1.878421 

Epoch 3
-------------------------------
loss: 1.911198  [   64/60000]
loss: 1.879103  [ 6464/60000]
loss: 1.754470  [12864/60000]
loss: 1.808783  [19264/60000]
loss: 1.698743  [25664/60000]
loss: 1.641341  [32064/60000]
loss: 1.692547  [38464/60000]
loss: 1.572325  [44864/60000]
loss: 1.615470  [51264/60000]
loss: 1.514065  [57664/60000]
Test Error: 
 Accuracy: 61.4%, Avg loss: 1.516057 

Epoch 4
-------------------------------
loss: 1.585148  [   64/60000]
loss: 1.546446  [ 6464/60000]
loss: 1.389959  [12864/60000]
loss: 1.469239  [19264/60000]
loss: 1.348574  [25664/60000]
loss: 1.338207  [32064/60000]
loss: 1.377912  [38464/60000]
loss: 1.281635  [44864/60000]
loss: 1.329263  [51264/60000]
loss: 1.231211  [57664/60000]
Test Error: 
 Accuracy: 64.0%, Avg loss: 1.246431 

Epoch 5
-------------------------------
loss: 1.323614  [   64/60000]
loss: 1.303683  [ 6464/60000]
loss: 1.134255  [12864/60000]
loss: 1.242207  [19264/60000]
loss: 1.116991  [25664/60000]
loss: 1.140264  [32064/60000]
loss: 1.183012  [38464/60000]
loss: 1.101487  [44864/60000]
loss: 1.150952  [51264/60000]
loss: 1.066533  [57664/60000]
Test Error: 
 Accuracy: 65.0%, Avg loss: 1.080341 

Done!
```

## 保存模型

保存模型的一种常用方法是序列化包含模型内部的状态字典（包含模型参数）。

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

```txt
Saved PyTorch Model State to model.pth
```

## 加载模型

加载模型包括：重新构建模型的结构，并将状态字典加载到模型：

```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

```txt
<All keys matched successfully>
```

然后就能用该模型来预测。

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

```txt
Predicted: 'Ankle boot', Actual: 'Ankle boot'
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
