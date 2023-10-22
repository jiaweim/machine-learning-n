# 快速入门

- [快速入门](#快速入门)
  - [简介](#简介)
  - [数据处理](#数据处理)
  - [创建模型](#创建模型)
  - [优化模型参数](#优化模型参数)
  - [保存模型](#保存模型)
  - [加载模型](#加载模型)
  - [参考](#参考)

Last updated: 2023-02-06, 10:31
****

## 简介

下面介绍机器学习中常见任务的 API。

## 数据处理

PyTorch 有两个处理数据的基础类：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。`Dataset` 包含样本及其标签，`DataLoader` 将 `Dataset` 包装为可迭代对象。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch 为不同领域提供了不同的库，如 [TorchText](https://pytorch.org/text/stable/index.html), [TorchVision](https://pytorch.org/vision/stable/index.html) 和 [TorchAudio](https://pytorch.org/audio/stable/index.html)，这些库都包含数据集。本教程将使用 TorchVision 的一个数据集。

`torchvision.datasets` 模块包含许多真实视觉数据的 `Dataset` 对象，如 CIFAR，COCO [等](https://pytorch.org/vision/stable/datasets.html)，下面以 FashionMNIST 数据集为例。每个 TorchVision `Dataset` 包含两个参数：`transform` 和 `target_transform`，分别用于修改样本和标签。

```python
# 下载训练数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
# 下载测试数据
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

将 `Dataset` 作为参数传递给 DataLoader，它在数据集上封装了一个迭代器，并支持自动 batching, sampling, shuffling 以及多进程加载。下面定义 batch size 为 64，即 dataloader 迭代器的每个元素包含 64 对样本和标签。

```python
batch_size = 64

# 创建 DataLoader
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
- 在 `forward` 函数中定义网络的前向传播。

为了加速神经网络计算，可以将模型移到 GPU。

```python
# 使用 CPU 或 GPU 训练模型
device = "cuda" if torch.cuda.is_available() else "cpu"
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
            nn.Linear(512, 10)
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

需要[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)和[优化器](https://pytorch.org/docs/stable/optim.html)来训练模型：

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在单个训练循环中，模型对训练数据（按 batch 输入）进行预测，并根据反向传播预测误差来调整模型参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算推理 error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

在测试集上检查模型性能。

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

训练过程进行几次迭代（epochs）。在每个 epoch，模型调整参数以做出更好的预测。我们在每个 epoch 结尾打印模型的精度和损失，期望精度随时间的推移增加，而损失则随时间的推移而减少。

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
loss: 2.298689  [    0/60000]
loss: 2.291410  [ 6400/60000]
loss: 2.276740  [12800/60000]
loss: 2.281389  [19200/60000]
loss: 2.250101  [25600/60000]
loss: 2.218611  [32000/60000]
loss: 2.239000  [38400/60000]
loss: 2.198720  [44800/60000]
loss: 2.201087  [51200/60000]
loss: 2.178073  [57600/60000]
Test Error: 
 Accuracy: 43.4%, Avg loss: 2.166579 

Epoch 2
-------------------------------
loss: 2.166195  [    0/60000]
loss: 2.160911  [ 6400/60000]
loss: 2.109460  [12800/60000]
loss: 2.137640  [19200/60000]
loss: 2.082038  [25600/60000]
loss: 2.017595  [32000/60000]
loss: 2.061009  [38400/60000]
loss: 1.977701  [44800/60000]
loss: 1.986521  [51200/60000]
loss: 1.923424  [57600/60000]
Test Error: 
 Accuracy: 57.7%, Avg loss: 1.914980 

Epoch 3
-------------------------------
loss: 1.932753  [    0/60000]
loss: 1.910626  [ 6400/60000]
loss: 1.798431  [12800/60000]
loss: 1.850233  [19200/60000]
loss: 1.737147  [25600/60000]
loss: 1.674929  [32000/60000]
loss: 1.711906  [38400/60000]
loss: 1.605008  [44800/60000]
loss: 1.633193  [51200/60000]
loss: 1.524648  [57600/60000]
Test Error: 
 Accuracy: 61.5%, Avg loss: 1.540279 

Epoch 4
-------------------------------
loss: 1.595178  [    0/60000]
loss: 1.563596  [ 6400/60000]
loss: 1.415254  [12800/60000]
loss: 1.494914  [19200/60000]
loss: 1.367562  [25600/60000]
loss: 1.355115  [32000/60000]
loss: 1.375423  [38400/60000]
loss: 1.296142  [44800/60000]
loss: 1.336275  [51200/60000]
loss: 1.227030  [57600/60000]
Test Error: 
 Accuracy: 63.4%, Avg loss: 1.258707 

Epoch 5
-------------------------------
loss: 1.329095  [    0/60000]
loss: 1.310024  [ 6400/60000]
loss: 1.150906  [12800/60000]
loss: 1.261737  [19200/60000]
loss: 1.129152  [25600/60000]
loss: 1.150464  [32000/60000]
loss: 1.173875  [38400/60000]
loss: 1.110014  [44800/60000]
loss: 1.154168  [51200/60000]
loss: 1.064011  [57600/60000]
Test Error: 
 Accuracy: 64.5%, Avg loss: 1.089973 

Done!
```

## 保存模型

保存模型的一种常用方法是序列化包含模型内部状态的字典（包含模型参数）。

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

```txt
Saved PyTorch Model State to model.pth
```

## 加载模型

加载模型包括重新创建模型，并将状态字典加载到模型：

```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
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
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: '{predicted}', Actual: '{actual}'")
```

```txt
Predicted: 'Ankle boot', Actual: 'Ankle boot'
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
