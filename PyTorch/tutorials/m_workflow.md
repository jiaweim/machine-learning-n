# PyTorch 模型构建流程

- [PyTorch 模型构建流程](#pytorch-模型构建流程)
  - [简介](#简介)
  - [准备数据](#准备数据)
  - [准备模型](#准备模型)
    - [训练](#训练)


## 简介

```python
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module
import torch.nn.functional as F  # 激活函数
```

PyTorch 模型结构：

- 继承 `torch.nn.Module`；
- `__init__()`：负责实例化 layers，并加载需要的其它数据（如 NLP 模型可能需要加载词汇表）；
- `forward()`：定义正向传播执行的所有计算；


## 准备数据

mini-batch size 一般选择 2 的指数，如 16, 32, 64, 128，**32** 最常用，可作为默认选项。

```python
from torch.utils.data.dataset import random_split

torch.manual_seed(13)

# 使用 numpy 数组创建张量
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

# 使用所有数据创建 dataset（只适合小数据）
dataset = TensorDataset(x_tensor, y_tensor)

# 拆分数据
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# 分别创建 DataLoader
train_loader = DataLoader(
    dataset=train_data, 
    batch_size=16, # 示例数据只有 100 个，所以 batch_size 设置小点
    shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)
```

## 准备模型

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 学习率
lr = 0.1

# 创建模型，并移到 device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# 定义 optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义 loss function
loss_fn = nn.MSELoss(reduction='mean')

# Creates the train_step function for our model, loss function and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Creates the val_step function for our model and loss function
val_step_fn = make_val_step_fn(model, loss_fn)

# Creates a Summary Writer to interface with TensorBoard
writer = SummaryWriter('runs/simple_linear_regression')

# Fetches a single mini-batch so we can use add_graph
x_sample, y_sample = next(iter(train_loader))
writer.add_graph(model, x_sample.to(device))
```

```python
def make_train_step_fn(model, loss_fn, optimizer):
    # 单步 train 函数
    def perform_train_step_fn(x, y):
        #设置 TRAIN 模式
        model.train()

        # 1. 模型预测
        yhat = model(x)
        # 2. 计算损失
        loss = loss_fn(yhat, y)
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return perform_train_step_fn
```

```python
def make_val_step_fn(model, loss_fn):
    # 单步 validation 函数
    def perform_val_step_fn(x, y):
        # 设置 EVAL 模式
        model.eval()

        # 1. 模型预测
        yhat = model(x)
        # 2. 计算损失
        loss = loss_fn(yhat, y)
        return loss.item()

    return perform_val_step_fn
```

### 训练

```python
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss
```

```python
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    # 训练
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # 验证
    # no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # Records both losses for each epoch under the main tag "loss"
    writer.add_scalars(main_tag='loss',
                       tag_scalar_dict={'training': loss, 'validation': val_loss},
                       global_step=epoch)

# Closes the writer
writer.close()
```



