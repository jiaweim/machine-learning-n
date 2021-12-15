# CIFAR

- [CIFAR](#cifar)
  - [简介](#简介)
  - [CIFAR-10](#cifar-10)
    - [下载 CIFAR-10](#下载-cifar-10)
  - [CIFAR-100](#cifar-100)
  - [参考](#参考)

2021-12-15, 14:50
***

## 简介

CIFAR-10 和 CIFAR-100 是一个包含 8,000 万张微型图像数据集的带标签子集。由 Alex Krizhevsky, Vinod Nair 和 Geoffrey Hinton 收集整理。

## CIFAR-10

CIFAR-10 数据集包含 60,000 张 32x32 彩色图片，分为 10 个类别，每个类别 6,000 张。分为训练图片 50,000 张，测试图片 10,000 张。

该数据集分为 5 个训练 batch 和 1 个测试 batch，每个 batch 包含 10,000 张图片。测试 batch 包含从每个类中随机选择的 1000 张图片。剩余的图片以随机的顺序出现在 5 个续联 batch 中。由于是随机的，所以不同 batch 中包含的不同类别图片数目可能不相等。

下图是数据集包含的 10 个类别，以及随机选择的该类别的 10 张图片：

![](images/2021-12-15-15-07-19.png)

这些类别是完全互斥的，即一张图片只属于一个类别。

### 下载 CIFAR-10

|版本|大小|md5sum|
|---|---|---|
|[CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)|163 MB|c58f30108f718f92721af3b95e74349a|
|[CIFAR-10 Matlab version](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz)|175 MB|70270af85842c9e89bb428ec9976c926|
|[CIFAR-10 binary version (适合 C 语言)](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)|162 MB|c32a1d4ab5d03f1284b67883e8d87530|

## CIFAR-100

和 CIFAR-10 一样，CIFAR-100 包含 100 个类别，每个类别 600 张图片，对应 500 张训练图片和 100 张测试图片。

CIFAR-100 的 100 个分类可以归为 20 个超类。每个图片带有一个细分类标签（所属类别）和粗分类标签（所属超类）。具体关系如下：

|超类|分类|
|---|---|
|aquatic mammals|beaver, dolphin, otter, seal, whale|
|fish|aquarium fish, flatfish, ray, shark, trout|
|flowers|orchids, poppies, roses, sunflowers, tulips|
|food containers|bottles, bowls, cans, cups, plates|
|fruit and vegetables|apples, mushrooms, oranges, pears, sweet peppers|
|household electrical devices|clock, computer keyboard, lamp, telephone, television|
|household furniture|bed, chair, couch, table, wardrobe|
|insects|bee, beetle, butterfly, caterpillar, cockroach|
|large carnivores|bear, leopard, lion, tiger, wolf|
|large man-made outdoor things|bridge, castle, house, road, skyscraper|
|large natural outdoor scenes|cloud, forest, mountain, plain, sea|
|large omnivores and herbivores|camel, cattle, chimpanzee, elephant, kangaroo|
|medium-sized mammals|fox, porcupine, possum, raccoon, skunk|
|non-insect invertebrates|crab, lobster, snail, spider, worm|
|people|baby, boy, girl, man, woman|
|reptiles|crocodile, dinosaur, lizard, snake, turtle|
|small mammals|hamster, mouse, rabbit, shrew, squirrel|
|trees|maple, oak, palm, pine, willow|
|vehicles 1|bicycle, bus, motorcycle, pickup truck, train|
|vehicles 2|lawn-mower, rocket, streetcar, tank, tractor|

## 参考

- https://www.cs.toronto.edu/~kriz/cifar.html
