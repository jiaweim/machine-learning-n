# 计算机视觉迁移学习

- [计算机视觉迁移学习](#计算机视觉迁移学习)
  - [简介](#简介)
  - [加载数据](#加载数据)
    - [可视化图像](#可视化图像)
  - [训练模型](#训练模型)
    - [可视化模型预测](#可视化模型预测)
  - [微调 convnet](#微调-convnet)
    - [训练和评估](#训练和评估)
  - [ConvNet 作为特征提取器](#convnet-作为特征提取器)
    - [训练和评估](#训练和评估-1)
  - [参考](#参考)

Last updated: 2022-12-19, 15:30
****

## 简介

下面介绍如何使用迁移学习训练卷积神经网络，以进行图像分类。

在实践中，很少有人从头开始训练整个卷积网络，因为获得一个足够大的数据集很难。相反，通常在一个非常大的数据集（如 ImageNet）进行预训练，然后将预训练网络用过感兴趣任务的初始化器或特征提取器。

迁移学习的两个主要应用场景：

- **微调 convnet**：不使用随机初始化，而是用预训练的网络来初始化。
- **ConvNet 作为特征提取器**：除了最后的全连接层，冻结网络余下的所有权重。仅训练最后一个全连接层。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()  # interactive mode
```

## 加载数据

下面使用 torchvision 和 torch.utils.data 加载数据。

问题：训练一个模型区分蚂蚁和蜜蜂。
数据：蚂蚁和蜜蜂各有大概 120 张训练图像和 75 张验证图像（ImageNet 的一个子集）。

如果从头开始训练，该数据集过于小，所以采用迁移学习。

[下载](https://download.pytorch.org/tutorial/hymenoptera_data.zip)数据，并提取到当前目录：

```python
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 可视化图像

下面可视化一些训练图像，以理解数据增强：

```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

## 训练模型

下面编写一个通用函数来训练模型，并解释：

- 学习率调整
- 保存最佳模型

下面参数 `scheduler` 指 `torch.optim.lr_scheduler` 中定义的对象。

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

### 可视化模型预测

显示少数图像预测结果的通用函数：

```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

## 微调 convnet

加载预训练模型并重置最后的全连接层：

```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

### 训练和评估

在 CPU 上需要大概 15-25 分钟，在 GPU 上不到一分钟：

```python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

```txt
Epoch 0/24
----------
train Loss: 0.6404 Acc: 0.6885
val Loss: 0.1729 Acc: 0.9150

Epoch 1/24
----------
train Loss: 0.4765 Acc: 0.8238
val Loss: 0.3751 Acc: 0.8693

Epoch 2/24
----------
train Loss: 0.5239 Acc: 0.8074
val Loss: 0.2883 Acc: 0.8824
...
...
...
Epoch 22/24
----------
train Loss: 0.2440 Acc: 0.9057
val Loss: 0.1870 Acc: 0.9608

Epoch 23/24
----------
train Loss: 0.2603 Acc: 0.8893
val Loss: 0.1872 Acc: 0.9542

Epoch 24/24
----------
train Loss: 0.3001 Acc: 0.8770
val Loss: 0.1801 Acc: 0.9477

Training complete in 2m 41s
Best val Acc: 0.960784
```

```python
visualize_model(model_ft)
```

![](2022-12-19-15-18-26.png)

## ConvNet 作为特征提取器

需要冻结最后一层之外的所有网络，设置 `requires_grad = False` 冻结参数。

```python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

### 训练和评估

在 CPU 上大概需要 10 min，因为大部分梯度不需要计算，不会前向计算依然需要

```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

```txt
Epoch 0/24
----------
train Loss: 0.6344 Acc: 0.6434
val Loss: 0.2390 Acc: 0.9216

Epoch 1/24
----------
train Loss: 0.6260 Acc: 0.7213
val Loss: 0.1956 Acc: 0.9346

Epoch 2/24
----------
train Loss: 0.6534 Acc: 0.7582
val Loss: 0.1977 Acc: 0.9281
...
...
Epoch 22/24
----------
train Loss: 0.3881 Acc: 0.8443
val Loss: 0.2322 Acc: 0.9216

Epoch 23/24
----------
train Loss: 0.3878 Acc: 0.8279
val Loss: 0.2221 Acc: 0.9281

Epoch 24/24
----------
train Loss: 0.2904 Acc: 0.8730
val Loss: 0.2027 Acc: 0.9412

Training complete in 2m 20s
Best val Acc: 0.947712
```

```python
visualize_model(model_conv)

plt.ioff()
plt.show()
```

![](2022-12-19-15-30-07.png)

## 参考

- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
