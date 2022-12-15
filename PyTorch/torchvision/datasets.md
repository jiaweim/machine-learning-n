# Datasets

## 简介

Torchvision 在 `torchvision.datasets` 模块提供了许多内置数据集，以及用于自定义数据集的工具类。

## 内置数据集

所有数据集都是 `torch.utils.data.Dataset` 的子类，实现了 `__getitem__` 和 `__len__` 方法。因此都可以传入 `torch.utils.data.DataLoader` 实现并行加载功能。例如：

```python
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(
    imagenet_data,
    batch_size=4,
    shuffle=True,
    num_workers=args.nThreads)
```

所有数据集都有类似的 API，都有两个共同参数：`transform` 和 `target_transform`，用来设置对输入和 target 的变换操作。

### Image classification

|数据集|说明|
|---|---|
|`Caltech101`(root[, target_type, transform, ...])|[Caltech 101](https://data.caltech.edu/records/20086) 数据集|
|`Caltech256`(root[, transform, ...])|[Caltech 256](https://data.caltech.edu/records/20087)数据集|

CelebA(root[, split, target_type, ...])

Large-scale CelebFaces Attributes (CelebA) Dataset Dataset.

CIFAR10(root[, train, transform, ...])

CIFAR10 Dataset.

CIFAR100(root[, train, transform, ...])

CIFAR100 Dataset.

Country211(root[, split, transform, ...])

The Country211 Data Set from OpenAI.

DTD(root[, split, partition, transform, ...])

Describable Textures Dataset (DTD).

EMNIST(root, split, **kwargs)

EMNIST Dataset.

EuroSAT(root[, transform, target_transform, ...])

RGB version of the EuroSAT Dataset.

FakeData([size, image_size, num_classes, ...])

A fake dataset that returns randomly generated images and returns them as PIL images

FashionMNIST(root[, train, transform, ...])

Fashion-MNIST Dataset.

FER2013(root[, split, transform, ...])

FER2013 Dataset.

FGVCAircraft(root[, split, ...])

FGVC Aircraft Dataset.

Flickr8k(root, ann_file[, transform, ...])

Flickr8k Entities Dataset.

Flickr30k(root, ann_file[, transform, ...])

Flickr30k Entities Dataset.

Flowers102(root[, split, transform, ...])

Oxford 102 Flower Dataset.

Food101(root[, split, transform, ...])

The Food-101 Data Set.

GTSRB(root[, split, transform, ...])

German Traffic Sign Recognition Benchmark (GTSRB) Dataset.

INaturalist(root[, version, target_type, ...])

iNaturalist Dataset.

ImageNet(root[, split])

ImageNet 2012 Classification Dataset.

KMNIST(root[, train, transform, ...])

Kuzushiji-MNIST Dataset.

LFWPeople(root[, split, image_set, ...])

LFW Dataset.

LSUN(root[, classes, transform, ...])

LSUN dataset.

MNIST(root[, train, transform, ...])

MNIST Dataset.

Omniglot(root[, background, transform, ...])

Omniglot Dataset.

OxfordIIITPet(root[, split, target_types, ...])

Oxford-IIIT Pet Dataset.

Places365(root, split, small, download, ...)

Places365 classification dataset.

PCAM(root[, split, transform, ...])

PCAM Dataset.

QMNIST(root[, what, compat, train])

QMNIST Dataset.

RenderedSST2(root[, split, transform, ...])

The Rendered SST2 Dataset.

SEMEION(root[, transform, target_transform, ...])

SEMEION Dataset.

SBU(root[, transform, target_transform, ...])

SBU Captioned Photo Dataset.

StanfordCars(root[, split, transform, ...])

Stanford Cars Dataset

STL10(root[, split, folds, transform, ...])

STL10 Dataset.

SUN397(root[, transform, target_transform, ...])

The SUN397 Data Set.

SVHN(root[, split, transform, ...])

SVHN Dataset.

USPS(root[, train, transform, ...])

USPS Dataset.

## 自定义数据集的基类

|类|说明|
|---|---|
DatasetFolder(root, loader[, extensions, ...])

A generic data loader.

ImageFolder(root, transform, ...)

A generic data loader where the images are arranged in this way by default: .

VisionDataset(root[, transforms, transform, ...])

Base Class For making datasets which are compatible with torchvision.

## 参考

- https://pytorch.org/vision/stable/datasets.html
