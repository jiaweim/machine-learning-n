# PyTorch 模型

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


## 流程 v1

- 准备数据

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 将数据转换为张量，然后移到指定 device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
```

- 准备模型

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
```

- 训练

训练循环中的代码，基本是固定的。

```python
n_epochs = 1000

for epoch in range(n_epochs):
    # 进入 TRAIN 模式
    model.train()

    # 1. 计算预测结果
    yhat = model(x_train_tensor)

    # 2. 计算损失
    loss = loss_fn(yhat, y_train_tensor)

    # 3. 计算梯度
    loss.backward()

    # 4. 更新参数
    optimizer.step()
    optimizer.zero_grad()
```