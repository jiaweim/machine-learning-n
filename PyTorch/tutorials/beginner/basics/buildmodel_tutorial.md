# 构建模型

## 简介

神经网络由执行数据操作的 layers/modules 组成。[torch.nn](https://pytorch.org/docs/stable/nn.html) 命名空间提供了构建神经网络所需的模块。PyTorch 中的所有 module 都继承自 [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)。神经网络是包含其它 modules 的 module。这种嵌套结构方便构建和管理复杂的体系结构。

下面构建一个用于 FashionMNIST 数据集的图像进行分类的神经网络。

```python
import torch
from torch import nn
```

## 训练设备

我们一般希望能在 GPU 等加速器上训练模型。先检查 `torch.cuda` 是否可用，如果不可用，则继续使用 CPU：

```python
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

```txt
Using cuda device
```

## 定义类

通过继承 `nn.Module` 定义神经网络，并在 `__init__` 中初始化神经网络层。`nn.Module` 的子类在 `forward` 发方法中实现对输入数据的操作：

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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
```

创建 `NeuralNetwork` 实例，移到 `device`，然后打印其结构：

```python
model = NeuralNetwork().to(device)
print(model)
```

```txt
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

将输入数据传入模型，它将执行模型的 `forward` 方法，以及一些后台操作。注意不要直接调用 `model.forward()`。

在输入上调用模型返回一个二维张量，其中 dim=0 对应样本，dim=1 对应每个样本在 10 个类别上的原始预测值。将原始预测值传入 `nn.Softmax` 模块获得预测概率：

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print(logits)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```txt
tensor([[-0.0655, -0.0840,  0.1080,  0.0545,  0.0570,  0.0271,  0.0586, -0.0124,
         -0.0350, -0.1483]], device='cuda:0', grad_fn=<AddmmBackward0>)
Predicted class: tensor([2], device='cuda:0')
```

## Model Layers

下面分解说明 FashionMNIST 模型中的 layer。为了便于说明，先生成一个 3 个 28*28 的小批量图像，看看在网络中传递时数据的变化。

```python
input_image = torch.rand(3, 28, 28)
print(input_image.size())
```

```txt
torch.Size([3, 28, 28])
```

### nn.Flatten

使用 `nn.Flatten` 将 2D 的 28*28 图像转换为 1D 的包含 784 像素值的数组（保留 dim=0 的 batch 维度）。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

```txt
torch.Size([3, 784])
```

### nn.Linear

线性层根据其 weights 和 biases 对输入应用线性变化。

```python
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

```txt
torch.Size([3, 20])
```

### nn.ReLU

非线性激活函数是在模型的输入和输出之间创建复杂映射的基础。它们一般应用在线性变换后，以引入非线性，帮助神经网络学习更复杂的现象。

这个模型，我们使用 nn.ReLU：

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```txt
Before ReLU: tensor([[ 0.1013, -0.0208, -0.1458,  0.5702, -0.2807,  0.1173,  0.1898, -0.3770,
          0.5245,  0.5818, -0.4242,  0.5056, -0.0922,  0.0574,  0.5201, -0.2360,
         -0.1200,  0.2744, -0.3400,  0.3640],
        [ 0.2326,  0.2826, -0.4901,  0.4925, -0.2246,  0.0521,  0.3052, -0.0919,
          0.1759,  0.0465, -0.5521,  0.7622,  0.1907,  0.1203,  0.6985, -0.2119,
          0.2683,  0.6890, -0.2460,  0.3739],
        [-0.0602,  0.1413, -0.2680,  0.8032, -0.4018, -0.0762,  0.0394, -0.1787,
         -0.1507,  0.2068, -0.6265,  0.5108, -0.1756,  0.0363,  0.5440, -0.1034,
         -0.2537,  0.5540, -0.3600,  0.0754]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.1013, 0.0000, 0.0000, 0.5702, 0.0000, 0.1173, 0.1898, 0.0000, 0.5245,
         0.5818, 0.0000, 0.5056, 0.0000, 0.0574, 0.5201, 0.0000, 0.0000, 0.2744,
         0.0000, 0.3640],
        [0.2326, 0.2826, 0.0000, 0.4925, 0.0000, 0.0521, 0.3052, 0.0000, 0.1759,
         0.0465, 0.0000, 0.7622, 0.1907, 0.1203, 0.6985, 0.0000, 0.2683, 0.6890,
         0.0000, 0.3739],
        [0.0000, 0.1413, 0.0000, 0.8032, 0.0000, 0.0000, 0.0394, 0.0000, 0.0000,
         0.2068, 0.0000, 0.5108, 0.0000, 0.0363, 0.5440, 0.0000, 0.0000, 0.5540,
         0.0000, 0.0754]], grad_fn=<ReluBackward0>)
```

### nn.Sequential

[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 是一个模块的有序容器。数据按照定义的顺序在模块中传递。使用 Sequential 容器可以迅速组合神经网络：

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
```

### nn.Softmax

神经网络最后一层返回的 `logits` 是 [-infty, infty] 之间的原始值，传递给 `nn.Softmax` 模块，`nn.Softmax` 将 `logits` 缩放到 [0, 1] 之间，表示模型对各个类的预测概率。`dim` 参数指示缩放的维度，缩放后该维度所有值的和必须为 1.

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 模型参数

神经网络中的许多 layer 都是参数化的，即具有相关的 weights 和 biases，并在训练过程中被优化。继承 `nn.Module` 将自动记录模块对象中定义的所有字段，可以使用 `parameters()` 或 `named_parameters()` 方法访问这些参数。

下面迭代每个参数，并输出其 size 和值：

```python
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
```

```txt
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values: tensor([[ 9.8009e-05, -3.4826e-02,  3.4859e-02,  ...,  2.5165e-02,
         -8.2530e-03,  3.5230e-02],
        [-9.4047e-03,  1.4986e-02,  3.4582e-02,  ...,  1.6275e-02,
         -1.7859e-02, -2.3725e-02]], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values: tensor([-0.0093, -0.0214], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values: tensor([[-0.0368, -0.0137,  0.0347,  ..., -0.0079,  0.0059, -0.0387],
        [ 0.0317,  0.0095,  0.0428,  ...,  0.0128, -0.0187, -0.0190]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values: tensor([-0.0352,  0.0198], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values: tensor([[ 0.0009, -0.0058,  0.0286,  ...,  0.0356, -0.0095, -0.0376],
        [-0.0223,  0.0272, -0.0054,  ..., -0.0049, -0.0029,  0.0209]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values: tensor([ 0.0231, -0.0391], device='cuda:0', grad_fn=<SliceBackward0>) 

```

## 参考

- https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
