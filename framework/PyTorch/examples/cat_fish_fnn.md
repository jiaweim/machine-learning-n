# 猫鱼图片分类

- [猫鱼图片分类](#猫鱼图片分类)
  - [数据](#数据)
  - [准备数据集](#准备数据集)
    - [创建训练集](#创建训练集)
    - [创建验证集和测试集](#创建验证集和测试集)
    - [创建 DataLoader](#创建-dataloader)
  - [定义网络](#定义网络)
  - [训练](#训练)
  - [预测](#预测)
  - [保存模型](#保存模型)

***

## 数据

图片下载地址：

https://drive.google.com/file/d/16h8E7dnj5TpxF_ex4vF2do20iMWziM70

该数据集分为三部分：

- train
  - cat (302)
  - fish (488)
- test
  - cat (100)
  - fish (94)
- val
  - cat (100)
  - fish (55)

## 准备数据集

`check_image` 传递给 `ImageFolder` 的 `is_valid_file` 参数，确保 PIL 能够打开图片。

```python
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False
```

### 创建训练集

设置转换器，对每个图像应该：

- 缩放到 64x64
- 转换为 Tensor
- 使用 ImageNet 的 mean 和 std 归一化图像

```python
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
```

创建训练集：

```python
train_data_path = "./train/"
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path,
    transform=img_transforms)
```

上面归一化使用的均值和方差来自 ImageNet 数据集，也可以专门计算这些鱼和猫数据集的均值和标准差，不过很多人都直接使用 ImageNet 常量。

> **NOTE** 上面将图像大小调整为 64x64，是随意选择，目的是为了让训练速度更快。大多数现有架构都使用 224x224 或 299x299 图像输入。一般来说，输入尺寸越大，网络学习的数据越多，但 GPU 内存只能容纳很小批量的图像。

### 创建验证集和测试集

要求验证集的图像不在训练集中出现。

```python
val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms)
```

创建测试集，在训练完后测试模型：

```python
test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms)
```

|数据集|说明|
|---|---|
|训练集|在训练过程中用来更新模型|
|验证集|评估模型在这个问题上的泛化能力，不用来直接更新模型|
|测试集|训练完成后，用来对模型的性能进行最后的评价|

### 创建 DataLoader

```python
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
```

## 定义网络

在 PyTorch 中定义神经网络，需要继承 `nn.Module`，下面定义一个三层全连接层，使用 ReLU 激活函数：

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50) # 隐藏层比较随意
        self.fc3 = nn.Linear(50, 2) # 输出类别为 2

    def forward(self, x):
        x = x.view(-1, 12288) # 转换为一维 Tensor 64*64*3
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

simplenet = SimpleNet()
```

创建优化器：

```python
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)
```

## 训练

根据是否有 GPU，选择设备：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

simplenet.to(device)
```

单纯的训练循环很简单：

- 每次取一个 batch 数据
- 计算模型对输入数据的结果
- 根据推理结果和期望输出计算损失
- 调用 `backward()` 计算梯度
- 调用 `optimizer.step()` 根据梯度调整权重

```python
for epoch in range(epochs):
    for batch in train_data_loader:
        optimizer.zero_grad()
        input, target = batch
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

由于训练方法比较通用，所以将训练代码单独创建一个函数：

```python
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train() # 开始训练
        for batch in train_loader:
            optimizer.zero_grad() # 梯度归零
            inputs, targets = batch
            inputs = inputs.to(device) # batch 数据移到 GPU/CPU
            targets = targets.to(device)
            output = model(inputs) # 计算输出
            loss = loss_fn(output, targets) # 计算损失
            loss.backward() # 计算梯度
            optimizer.step() # 更新权重
            training_loss += loss.data.item() * inputs.size(0) 
        training_loss /= len(train_loader.dataset) #总损失均值

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                valid_loss, num_correct / num_examples))
```

提供参数，开始训练：

```python
train(simplenet,
    optimizer,
    torch.nn.CrossEntropyLoss(),
    train_data_loader,
    val_data_loader,
    epochs=5,
    device=device)
```

## 预测

训练完成后，就可以使用网络对新的图片进行推理，判断是鱼还是猫：

```python
labels = ['cat', 'fish']

img = Image.open("./val/fish/100_1422.JPG") # 加载图片
img = img_transforms(img).to(device) # 执行图像转变
img = torch.unsqueeze(img, 0) # 插入 batch 维度

simplenet.eval()
prediction = F.softmax(simplenet(img), dim=1) # 转换为概率
prediction = prediction.argmax() # 获得概率最大的索引
print(labels[prediction]) # 输出索引对应类别
```

## 保存模型

保存模型结构和参数：

```python
torch.save(simplenet, "/tmp/simplenet")
```

重新加载模型：

```python
simplenet = torch.load("/tmp/simplenet")
```

只保存参数：

```python
torch.save(simplenet.state_dict(), "/tmp/simplenet")
```

加载参数：

```python
simplenet = SimpleNet()
simplenet_state_dict = torch.load("/tmp/simplenet")
simplenet.load_state_dict(simplenet_state_dict)   
```

只保存参数的好处是，如果模型结构发生了变化，在 `load_state_dict` 时可以设置 `strict=False`，为 state_dict 中确实有的 layer 指定相应参数，如果所加载的 state_dict 与模型当前结构相比缺少或增加了某些层，也不会报错。
