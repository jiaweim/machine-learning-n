# 聚类分析快速入门

## 简介

聚类分析（cluster analysis），一种无监督学习技术。聚类的目的是在数据中找到一种分组方案，使得来自相同 cluster 的对象比来自不同 cluster 的对象更相似。

下面介绍如下内容：

- 使用流行的 k-means 算法寻找相似中心
- 采用 bottom-up 方法构建分层聚类树
- 使用基于密度的聚类方法鉴定任意形状的对象

## k-means 聚类

k-means 是最流行的聚类算法之一，在学术界和工业界都有着广泛应用。聚类（clustering）或聚类分析（cluster analysis）是一种根据对象相似性进行分组的技术。聚类的面向业务的应用包括按照主题对文档、音乐或电影进行分组，基于客户的购买行为进行分组，作为推荐系统的基础。

### k-means 聚类的 scikit-learn 实现

k-means 算法很容易实现，与其它算法相比，其计算性能高。k-means 属于基于原型的聚类算法（**prototype-based** clustering）。另外还有分层聚类（hierarchical）和基于密度（density）的聚类。

基于原型（prototype）的聚类，指每个 cluster 由一个原型来表示。原型，在连续特征中一般是相似点的质心（均值），在离散特征中一般是 medoid（最具代表性的点，能最小化相同 cluster 中与其它点的距离）。

k-means 擅长识别**球形** cluster，其缺点是需要提前指定 cluster 的数量 $k$。k 值的选择影响聚类效果。后面会介绍 **elbow** 和 **silhouette plots** 两个评价聚类效果的技术，可以辅助确定 cluster 的最优数量。

k-means 聚类可以应用于高维数据，不过为了方便可视化，下面所以二维数据为例：

```python

```


