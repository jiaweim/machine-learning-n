# 优化模型参数

- [优化模型参数](#优化模型参数)
  - [简介](#简介)
  - [预定义代码](#预定义代码)
  - [超参数](#超参数)
  - [优化循环](#优化循环)
    - [损失函数](#损失函数)
    - [优化器](#优化器)
  - [完整实现](#完整实现)
  - [参考](#参考)

Last updated: 2022-11-08, 15:54
****

## 简介

有了模型和数据后，下一步是使用数据训练、验证和测试模型。训练模型是一个迭代过程，每次迭代（称为一个 epoch）模型根据当前参数进行预测，计算预测值和实际值的误差（loss），收集误差对模型参数的导数，并使用梯度下降优化这些参数。

## 预定义代码

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root=r"D:\data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root=r"D:\data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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


model = NeuralNetwork()
```

## 超参数

超参数是可调整参数，用于控制模型的优化过程。超参数值影响模型的训练和收敛速度，更过信息可参考[超参数调整](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)。

为训练定义如下超参数：

- **Epoch 数**，在数据集上迭代的次数
- **Batch Size**，在更新参数前通过网络传播的数据样本数
- **Learning Rate**，每个 batch/epoch 后更新模型参数的幅度。过小的值会使学习速率缓慢，过大则可能导致训练不收敛。

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 优化循环

设置好超参数后，通过优化循环来训练和优化模型。优化循环的每次迭代称为一个 **epoch**。

每个 epoch 主要包含两部分：

- **训练循环**，迭代训练数据集，并尝试收敛到最佳参数
- **验证/测试循环**，迭代测试数据集，检查模型性能是否在改进

先简单介绍一下训练循环中使用到的一些概念。

### 损失函数

给模型喂入一些训练数据，未经训练的网络大概率不会给出正确的答案。**损失函数（loss function）** 度量模型预测结果与真实结果的差异程度，训练模型就是要最小化损失函数的值。为了计算损失，对给定输入样本数据进行预测，并将预测值与真实数据标签值进行比较。

常见的损失函数包括用于回归任务的均方差 [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)、用于分类任务的负对数似然 [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)。[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 结合 `nn.LogSoftmax` 和 `nn.NLLLoss`。

将模型输出的 logits 传入 `nn.CrossEntropyLoss`，这将归一化 logits，并计算预测误差。

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### 优化器

优化是在每个训练步骤中调整模型参数以减小模型误差的过程。**优化算法**定义了如何执行这个过程。所有优化逻辑封装在 `optimizer` 对象中。这里使用 SGD 优化器，PyTorch 还提供了许多[其它优化器](https://pytorch.org/docs/stable/optim.html)，如 ASAM、RMSProp 等，应用于不同类型的模型或数据。

使用模型待训练的参数初始化优化器，并传入超参数学习率。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练循环中，优化分三步进行：

- 调用 `optimizer.zero_grad()` 来重置模型参数的梯度。梯度默认累加，为了防止重复计算，在每次迭代时都要显式归零。
- 调用 `loss.backward()` 反向传播计算梯度。PyTorch 存储每个参数的梯度。
- 有了梯度，就可以调用 `optimizer.step()` 通过反向传播收集到的梯度调整参数。

## 完整实现

定义 `train_loop` 实现训练循环，`test_loop` 实现测试循环。

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

初始化损失函数和优化器，传入 `train_loop` 和 `test_loop`。可随机增加 epoch 数，查看模型的性能改进过程。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

```txt
Epoch 1
-------------------------------
loss: 2.295453  [    0/60000]
loss: 2.291088  [ 6400/60000]
loss: 2.271296  [12800/60000]
loss: 2.267435  [19200/60000]
loss: 2.252776  [25600/60000]
loss: 2.210875  [32000/60000]
loss: 2.230757  [38400/60000]
loss: 2.194520  [44800/60000]
loss: 2.186396  [51200/60000]
loss: 2.150742  [57600/60000]
Test Error: 
 Accuracy: 29.4%, Avg loss: 2.148574 

Epoch 2
-------------------------------
loss: 2.156786  [    0/60000]
loss: 2.151252  [ 6400/60000]
loss: 2.093557  [12800/60000]
loss: 2.110151  [19200/60000]
loss: 2.073736  [25600/60000]
loss: 1.995176  [32000/60000]
loss: 2.030679  [38400/60000]
loss: 1.951633  [44800/60000]
loss: 1.957083  [51200/60000]
loss: 1.876364  [57600/60000]
Test Error: 
 Accuracy: 57.7%, Avg loss: 1.878447 

Epoch 3
-------------------------------
loss: 1.909636  [    0/60000]
loss: 1.882801  [ 6400/60000]
loss: 1.764005  [12800/60000]
loss: 1.801928  [19200/60000]
loss: 1.717762  [25600/60000]
loss: 1.646998  [32000/60000]
loss: 1.671499  [38400/60000]
loss: 1.572597  [44800/60000]
loss: 1.601334  [51200/60000]
loss: 1.488696  [57600/60000]
Test Error: 
 Accuracy: 61.9%, Avg loss: 1.508771 

Epoch 4
-------------------------------
loss: 1.573309  [    0/60000]
loss: 1.541056  [ 6400/60000]
loss: 1.389572  [12800/60000]
loss: 1.456659  [19200/60000]
loss: 1.367432  [25600/60000]
loss: 1.343778  [32000/60000]
loss: 1.362260  [38400/60000]
loss: 1.285288  [44800/60000]
loss: 1.323994  [51200/60000]
loss: 1.223675  [57600/60000]
Test Error: 
 Accuracy: 63.5%, Avg loss: 1.246784 

Epoch 5
-------------------------------
loss: 1.320280  [    0/60000]
loss: 1.305915  [ 6400/60000]
loss: 1.138686  [12800/60000]
loss: 1.238160  [19200/60000]
loss: 1.140406  [25600/60000]
loss: 1.150574  [32000/60000]
loss: 1.175766  [38400/60000]
loss: 1.109631  [44800/60000]
loss: 1.149357  [51200/60000]
loss: 1.069791  [57600/60000]
Test Error: 
 Accuracy: 64.7%, Avg loss: 1.085315 

Epoch 6
-------------------------------
loss: 1.151206  [    0/60000]
loss: 1.159090  [ 6400/60000]
loss: 0.975790  [12800/60000]
loss: 1.103560  [19200/60000]
loss: 1.000942  [25600/60000]
loss: 1.020172  [32000/60000]
loss: 1.058834  [38400/60000]
loss: 0.997025  [44800/60000]
loss: 1.034579  [51200/60000]
loss: 0.972582  [57600/60000]
Test Error: 
 Accuracy: 65.5%, Avg loss: 0.980820 

Epoch 7
-------------------------------
loss: 1.033339  [    0/60000]
loss: 1.063290  [ 6400/60000]
loss: 0.864483  [12800/60000]
loss: 1.014192  [19200/60000]
loss: 0.913105  [25600/60000]
loss: 0.927679  [32000/60000]
loss: 0.980820  [38400/60000]
loss: 0.923671  [44800/60000]
loss: 0.954718  [51200/60000]
loss: 0.907045  [57600/60000]
Test Error: 
 Accuracy: 66.5%, Avg loss: 0.909541 

Epoch 8
-------------------------------
loss: 0.946621  [    0/60000]
loss: 0.996719  [ 6400/60000]
loss: 0.785050  [12800/60000]
loss: 0.951638  [19200/60000]
loss: 0.854988  [25600/60000]
loss: 0.859903  [32000/60000]
loss: 0.925325  [38400/60000]
loss: 0.874638  [44800/60000]
loss: 0.897937  [51200/60000]
loss: 0.859900  [57600/60000]
Test Error: 
 Accuracy: 67.8%, Avg loss: 0.858512 

Epoch 9
-------------------------------
loss: 0.880340  [    0/60000]
loss: 0.947012  [ 6400/60000]
loss: 0.725731  [12800/60000]
loss: 0.905893  [19200/60000]
loss: 0.813572  [25600/60000]
loss: 0.808931  [32000/60000]
loss: 0.883029  [38400/60000]
loss: 0.840396  [44800/60000]
loss: 0.856053  [51200/60000]
loss: 0.823870  [57600/60000]
Test Error: 
 Accuracy: 68.8%, Avg loss: 0.819949 

Epoch 10
-------------------------------
loss: 0.827493  [    0/60000]
loss: 0.907462  [ 6400/60000]
loss: 0.679529  [12800/60000]
loss: 0.870930  [19200/60000]
loss: 0.781898  [25600/60000]
loss: 0.769649  [32000/60000]
loss: 0.848787  [38400/60000]
loss: 0.815175  [44800/60000]
loss: 0.823978  [51200/60000]
loss: 0.794946  [57600/60000]
Test Error: 
 Accuracy: 70.1%, Avg loss: 0.789316 

Done!
```

## 参考

- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
