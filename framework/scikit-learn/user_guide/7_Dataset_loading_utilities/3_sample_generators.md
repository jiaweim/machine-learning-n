# 生成数据集

scikit-learn 包含生成各种随机样本的方法，可用于生成指定大小和复杂度的数据集。

## 用于 classification 和 clustering

这类 generator 生成一个 feature 矩阵及相应的离散 target。

### 单个 label

`make_blobs` 和 `make_classification` 通过为每个类别分配一个或多个正态分布的数据点 cluster 来创建 multiclass 数据集：

- `make_blobs` 可以控制每个 cluster 的中心和标准差，用于演示聚类
- `make_classification` 提供多种引入噪音的方式：相关、冗余和引入非信息特征；每个 class 有多个高斯 clusters；特征空间的线性变换。



## 用于 regression

## 用于 manifold learning

## 用于 decomposition