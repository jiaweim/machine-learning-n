# DJL 入门教程

## 创建第一个神经网络

### Step1 配置环境

这个教程需要安装 Java Jupyter Kernel。关于安装 kernel，可参考 [Jupyter notebooks](https://docs.djl.ai/master/docs/demos/jupyter/index.html)。

```sh
// Add the snapshot repository to get the DJL snapshot artifacts
// %mavenRepo snapshots https://central.sonatype.com/repository/maven-snapshots/

// 添加 Maven 依赖项
%maven ai.djl:api:0.28.0
%maven org.slf4j:slf4j-simple:1.7.36
```

```java
import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;
```

#### 神经网络

神经网络是一个黑箱函数。我们无需自己编写该函数，而是为其提供许多示例输出/输出对。然后训练网络以近似从输入到输出的函数。更多数据和更好的模型可以更准确地逼近真实函数。

#### 应用

在尝试构建神经网络时，就像构建大多数函数一样，首先要弄清楚的是函数签名。输入和输出类型是什么？使用相对一致签名的神经网络，我们将其称为[应用](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/Application.html)。

在本教程，我们将重点关注图像分类。这是最常见的应用之一，在深度学习中有悠久历史。在图像分类中，输入是单个图像，输出为图像类别。

```java
Application application = Application.CV.IMAGE_CLASSIFICATION;
```

#### 数据集

确定好应用后，接下来需要收集训练数据，并将其整理成数据集。收集和清理数据是深度学习最麻烦的任务。

使用的数据集可以是从各种来源收集的自定义数据，也可以使用在线免费数据集。自定义数据集可能更适合你的应用，免费数据集则更便捷。

**MNIST**

下面将使用 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 数据集，这是一个手写数字数据集。每个图像是一个大小为 28x28 的 0-9 黑白数字。通常用于深度学习入门，因为它体积小且训练速度快。

![Mnist Image](./images/MnistExamples.png)

理解数据集后，可以创建一个 `Dataset` 类的实现。这里使用内置的 MNIST 数据。

#### 多层感知机

有了数据集，下面可以选择一个模型来训练。本教程构建最简单的深度学习网络之一：**多层感知机**（Multilayer Perceptron, MLP）。

MLP 分层管理。第一层是输入层，包含输入数据；最后一次是输出层，产生网络的最终结果；两者之间称为隐藏层（hidden layer）。隐藏层越多、越大，MLP 就能够表示更复杂的函数。

下面的示例神经网络的输入大小为 3，一个大小为 3 的隐藏层，输出大小为 2。隐藏层的数量和大小通常通过实验确定。每对 layers 之间为线性操作（又称为 full-connected 操作）。每次线性操作后还有一个非线性激活函数（图中没有展示）。

<img src="./images/MultiLayerNeuralNetworkBigger_english.png" alt="MLP Image" width="500" />

### Step 2 确定输入和输出大小

MLP 模型的输入和输出为一维向量。你应该根据输入数据以及模型输出的用途来确定该向量的大小。

这里输入向量大小为 `28x28`，因为 MNIST 输入图像的高度和宽度均为 28，每个像素只需一个数字表示。对彩色图像，需要再乘以 3，对应 RGB 通道。

这里输出向量大小为 10，因为所有图像有 10 个可能的类别（0-9）。

```java
long inputSize = 28*28;
long outputSize = 10;
```

### Step 3 创建 SequentialBlock

#### NDArray

用于深度学习的核心数据类型是 `NDArray`。NDArray 表示多维、固定大小的同构数组。其行为与 Numpy python 包非常相似，但增加高效计算功能。还有一个辅助类 `NDList`，它是 `NDArray` list。

#### Block API

在 DJL 中，`Block`的功能类似于将输入 `NDList` 转换为输出 `NDList`。它们可以表示神经网络一部分的单个操作，也可以表示整个神经网络。`Block` 的特别之处在于它们包含许多参数，这些参数在深度学习过程中被训练。随着参数的训练，blocks 代表的函数变得越来越准确。

构建这些 block 的最简单方法是组合。与通过调用其它函数来构建函数类似，可以通过组合其它 blocks 来构建 block。将外部的 block 称为 parent，内部的 sub-blocks 称为 children。

DJL 提供了集中常见的 blocks 组合。对 MLP 使用 `SequentialBlock`，这是一个容器 block，其 children 串行排列，每个 child-block 的输出作为下一个 child-block 的输入。

```java
SequentialBlock block = new SequentialBlock();
```

### Step 4 向 SequentialBlock 添加 blocks

MLP 分为多层。每层包含一个线性的 `Linear` block 和非线性的激活函数。如果只有两个线性 blocks，其功能与单个组合线性相同 $f(x)=W_2(W_1x)=(W_2W_1)x=W_{combined}x$。激活函数用于线性 block 之间，从而能够表示非线性函数。下面使用流行的 ReLU 激活函数。

第一层和最后一层大小固定，取决于所需的输入和输出。网络中间层的数量和尺寸则可以自由调整。下面创建一个较小的 MLP，中间有两层，逐渐减少尺寸。通常，我们会尝试不同值，看看哪种参数最适合数据集。

```java
block.add(Blocks.batchFlattenBlock(inputSize));
block.add(Linear.builder().setUnits(128).build());
block.add(Activation::relu);
block.add(Linear.builder().setUnits(64).build());
block.add(Activation::relu);
block.add(Linear.builder().setUnits(outputSize).build());

block
```

```
SequentialBlock {
    batchFlatten
    Linear
    LambdaBlock
    Linear
    LambdaBlock
    Linear
}
```

到这里，就成功创建了第一个神经网络，完整 java 代码：

```java
Application application = Application.CV.IMAGE_CLASSIFICATION;
long inputSize = 28 * 28;
long outputSize = 10;

SequentialBlock block = new SequentialBlock();
block.add(Blocks.batchFlattenBlock(inputSize));
block.add(Linear.builder().setUnits(128).build());
block.add(Activation::relu);
block.add(Linear.builder().setUnits(64).build());
block.add(Activation::relu);
block.add(Linear.builder().setUnits(outputSize).build());

System.out.println(block);
```

## 训练第一个模型

下面介绍如何训练手写数字图像分类模型。

```sh
// Add the snapshot repository to get the DJL snapshot artifacts
// %mavenRepo snapshots https://central.sonatype.com/repository/maven-snapshots/

// 添加 maven 依赖项
%maven ai.djl:api:0.28.0
%maven ai.djl:basicdataset:0.28.0
%maven ai.djl:model-zoo:0.28.0
%maven ai.djl.mxnet:mxnet-engine:0.28.0
%maven org.slf4j:slf4j-simple:1.7.36
```

```java
import java.nio.file.*;

import ai.djl.*;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.types.*;
import ai.djl.training.*;
import ai.djl.training.dataset.*;
import ai.djl.training.initializer.*;
import ai.djl.training.loss.*;
import ai.djl.training.listener.*;
import ai.djl.training.evaluator.*;
import ai.djl.training.optimizer.*;
import ai.djl.training.util.*;
import ai.djl.basicmodelzoo.cv.classification.*;
import ai.djl.basicmodelzoo.basic.*;
```

### Step 1 准备 MNIST 数据集

为了训练模型，需要创建一个 `Dataset` 对象来包含训练数据。数据集是神经网络所表示函数的样本输入/输出对的集合。每个 输入/输出 表示为 `Record`。每个 record 可以有多种输入和输出数组。

有数数据学习是高度并行化的，因为训练通常不是一次使用一个 record，而是批量进行的 `Batch`。这可以显著提供性能，尤其是在处理图像时。

#### Sampler

接下来需要确定从数据集加载数据的参数。对 MNIST 需要的唯一参数是 `Sampler` 的选择。sampler 确定每次迭代数据时 batch 的大小和划分。这里对每个 batch 随机打乱数据，使用 batchSize 32。batchSize 通常是适合内存的最大 2 的指数。

```java
int batchSize = 32;
Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
mnist.prepare(new ProgressBar());
```

这里需要添加一个 Maven 依赖项：

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>basicdataset</artifactId>
    <version>0.35.0</version>
</dependency>
```

### Step 2 创建模型

接下来构建模型。`Model` 包含一个神经网络 `Block` 以及用于训练的其它组成。它包含有关输入、输出、shapes、数据类型等信息。一般在完成 `Block` 后，就需要使用 `Model` 了。

前面已经介绍如何构建神经网络，下面直接使用 model-zoo 内置的 MLP 模块。使用 model-zoo 需要添加一个依赖项：

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>model-zoo</artifactId>
    <version>0.35.0</version>
</dependency>
```

由于 MNIST 数据集中的图像是 28x28 灰度图像，因此我们创建一个输入微 28x28 的 MLP block。输出大小为 10。对隐藏层，通过尝试不同的值，最终选择 `new int[]{128, 64}`。

```java
Model model = Model.newInstance("mlp");
model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
```

### Step 3 创建 Trainer

`Trainer` 用于训练模型，通常使用 try-with 语法打开，在训练完成后关闭。

trainer 采用现有模型并尝试优化模型内的参数以匹配数据集。大多数优化基于[随机梯度下降](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)（SGD）。

#### 配置训练参数

在创建 `Trainer` 之前，需要一个 `TrainingConfig` 描述如何训练模型。

以下是一些常见的参数：

- **必需** `Loss` 函数：损失函数用于衡量模型与数据集的匹配程度。由于该函数的值越小越好，所以称为损失函数（loss）。`Loss` 是唯一必需设置的参数
- `Evaluator` 函数：评估函数也是用于衡量模型与数据集的匹配程度。与损失函数不同的是，它们只是供人查看，并不用于优化模型。由于很多损失函数不直观，因此添加其它 evaluators，如准确性，以帮助人们了解模型的性能。
- `TrainingListener`：`TrainingListener` 通过监听器接口向训练过程添加额外功能。如显示训练进度、在训练变得不明确时提前停止、记录性能指标。DJL 提供了几个默认的监听器。

另外还有 Device、Initializer、Optimizer 等选项，具体可参考 `TrainingConfig` 接口。

```java
DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
        .addEvaluator(new Accuracy())
        .addTrainingListeners(TrainingListener.Defaults.logging());

// 配置好训练参数，就可以为模型创建一个 trainer
Trainer trainer = model.newTrainer(config);
```

> [!NOTE]
>
> softmaxCrossEntropy 是一个用于分类问题的标准损失函数

### Step 4 初始化训练

在训练模型之前，需要初始化所有参数。可以使用 trainer 初始化参数，只需要传入输入 shape：

- 输入 shape 的第一个维度是 batch-size，不影响参数初始化，所以这里使用 1
- 输入 shape 的第二个维度是 MLP 的输入 shape，这里是输入图像的像素数

```java
trainer.initialize(new Shape(1, 28 * 28));
```

### Step 5 训练模型

可以训练模型了。

训练通常按 epoch 进行，一个 epoch 表示对数据集的每个样本训练模型一次，这比随机训练稍微快一点。

这里使用 `EasyTrain` 执行训练：

```java
int epoch = 2;
EasyTrain.fit(trainer, epoch, mnist, null);
```

```
19:19:06 [main] INFO a.d.t.l.LoggingTrainingListener[167] - Load MXNet Engine Version 1.9.0 in 0.036 ms.
Training:    100% |========================================| 
19:19:11 [main] INFO a.d.t.l.LoggingTrainingListener[67] - Epoch 1 finished.
Training:    100% |========================================| 
19:19:15 [main] INFO a.d.t.l.LoggingTrainingListener[67] - Epoch 2 finished.
```

这类要成功执行，还需要添加引擎，如 mxnet：

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-engine</artifactId>
    <version>0.35.0</version>
    <scope>runtime</scope>
</dependency>
```

### Step 6 保存模型

模型训练完成后可以保存它，以便后续重新加载。还可以添加元数据，如训练准确性、训练 epoch 等。

```java
Path modelDir = Paths.get("build/mlp");
Files.createDirectories(modelDir);
model.setProperty("Epoch", String.valueOf(epoch));
model.save(modelDir, "mlp");
System.out.println(model);
```

```
Model (
	Name: mlp
	Model location: C:\repositories\machine-learning-n\build\mlp
	Data Type: float32
	Epoch: 2
)
```

到这里就成功训练一个可以识别手写数字的模型。下面介绍如何使用模型分类图像。

本里完整代码在 [TrainMnist](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainMnist.java)。

## 使用模型推理

```sh
// Add the snapshot repository to get the DJL snapshot artifacts
// %mavenRepo snapshots https://central.sonatype.com/repository/maven-snapshots/

// 添加 Maven 依赖项
%maven ai.djl:api:0.28.0
%maven ai.djl:model-zoo:0.28.0
%maven ai.djl.mxnet:mxnet-engine:0.28.0
%maven ai.djl.mxnet:mxnet-model-zoo:0.28.0
%maven org.slf4j:slf4j-simple:1.7.36
```

```java
import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.ndarray.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.*;
```

### Step 1 加载手写数字图像

```java
Path imageFile = Paths.get("0.png");
Image img = ImageFactory.getInstance().fromFile(imageFile);
```

### Step 2 加载模型

加载模型进行推理。上一节将模型保存在 `build/mlp` 目录。

```java
Path modelDir = Paths.get("build/mlp");
Model model = Model.newInstance("mlp");
model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));
model.load(modelDir);
```

除了加载本地模型，还可以在 [model-zoo](https://docs.djl.ai/master/docs/model-zoo.html) 找到一些预训练模型。

### Step 3 创建 Translator



## 参考 

- https://docs.djl.ai/master/docs/demos/jupyter/tutorial/