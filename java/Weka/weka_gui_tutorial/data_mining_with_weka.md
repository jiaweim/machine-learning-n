# Weka 数据挖掘

## 绪论

Weka，全称 Waikato Environment for Knowledge Analysis，一个用于数据挖掘的免费开源工具。

参考书：Witten, I.H., Frank, E. & Hall, M.A. (2011) 
Data Mining: Practical Machine Learning Tools and Techniques, Third Edition. Morgan Kaufmann.

## 1. Weka 入门

### 1.1 简介

**Data Mining vs. Machine Learning**

通常认为，数据挖掘偏重应用，而机器学习侧重于算法。

Weka 包含：

- 100+ 分类算法
- 75 个数据预处理工具
- 25 个辅助特征选择工具
- 20 个聚类、相关性分析等大量经典算法

需要学习的内容：

- 如何加载数据
- 使用 filters 预处理数据
- 可视化分析
- 分类算法
- 解释结果
- 评估方法
- 了解不同模型
- 理解不同机器学习算法的原理
- 了解数据挖掘的缺陷

### 1.2 Explorer 界面

安装 Weka：省略。

Weka 有 4 个用户界面，如下：

<img src="./images/image-20250102100302699.png" alt="image-20250102100302699" style="zoom:50%;" />

下面**只需要用 Explorer 界面**。

- Experimenter 用于不同算法、不同数据集的性能比较
- KnowledgeFlow 用于构建算法流程
- Slimple CLI 是命令行界面

Explorer 界面如下：

<img src="./images/image-20250102100544506.png" alt="image-20250102100544506" style="zoom:50%;" />

下面主要介绍使用 Preprocess 预处理数据，用 Classify 进行数据分类，用 Visualize 进行可视化。

以天气数据集为例：

- 包含 14 个样本，对应 14 天的天气
- 每个样本包含 5 个与天气相关的属性
- Play 表示天气是否适合出去玩

目的：通过其它属性预测 Play 属性。

<img src="./images/image-20250102100749164.png" alt="image-20250102100749164" style="zoom: 50%;" />

打开数据集：

<img src="./images/image-20250102101036156.png" alt="image-20250102101036156" style="zoom:50%;" />

视图：

<img src="./images/image-20250102101201691.png" alt="image-20250102101201691" style="zoom:50%;" />

从这里可以查看不同属性和它们的取值分布。

另外，点击右上角的 **Edit** 可以编辑数据，编辑完成后点击 **Save...** 可保存数据。

### 1.3 数据集

<img src="./images/image-20250102102518765.png" alt="image-20250102102518765" style="zoom:50%;" />

该数据集最后一个为**类别**，是需要预测的值。

打开数据集，界面如下：

<img src="./images/image-20250102102846281.png" alt="image-20250102102846281" style="zoom:50%;" />

可以查看：

- 样本数（Instances）
- 属性数（Attributes）
- 默认最后一个属性为类别属性（class）
- 在直方图上方可以修改类别属性

Weka 将一个数据集称为 **Relation**。

分类任务要素：

- 数据集为标记数据
- 特征值分为离散（nominal）和连续（numeric）类型
- 预测问题也分为离散（classification）和连续（regression）

<img src="./images/image-20250102103558734.png" alt="image-20250102103558734" style="zoom: 33%;" />

打开天气数据集的 numeric 版本：

<img src="./images/image-20250102103832921.png" alt="image-20250102103832921" style="zoom:50%;" />

点击 **Edit**，可以发现和 weather.nomial.arff 的主要区别在于，温度和湿度变为数值：

<img src="./images/image-20250102103948182.png" alt="image-20250102103948182" style="zoom:50%;" />

在 Weka 界面可以查看数值数据的特征：

- 最小值
- 最大值
- 平均值
- 标准差
- 直方图分布

<img src="./images/image-20250102104116178.png" alt="image-20250102104116178" style="zoom:50%;" />

再来看 glass.arff 数据集，它是一个稍微大一点的数据集：

- 包含 214 instances
- 10 attributes

<img src="./images/image-20250102104200633.png" alt="image-20250102104200633" style="zoom:50%;" />

点击最后一个属性 **Type**，这是默认 class：

- Type 共有 7 种可能值，对应 7 种不同的玻璃类型
- 其它属性代表玻璃的不同特点

<img src="./images/image-20250102104351320.png" alt="image-20250102104351320" style="zoom:50%;" />

> [!IMPORTANT]
>
> 通过面板仔细检查各个属性，判断数据是否合理。获得高质量数据集是构建高质量模型的关键步骤。

### 1.4 构建 Classifier

**目标**：对 glass.arff 数据集构建一个 J48 分类器。

步骤：

- 打开 glass.arff 数据集
- 点击 Classify 面板
- 选择 J48 决策树模型
- 点击 **Start** 运行

<img src="./images/image-20250102105813643.png" alt="image-20250102105813643" style="zoom:50%;" />

在 Classifier output 可以查看输出，包括：

- 数据集信息
- 获得的模型信息
- 模型性能

采用默认参数，J48 的准确度为 66.8224 %。最后为混淆矩阵(confusion matrix)：

```
  a  b  c  d  e  f  g   <-- classified as
 50 15  3  0  0  1  1 |  a = build wind float
 16 47  6  0  2  3  2 |  b = build wind non-float
  5  5  6  0  0  1  0 |  c = vehic wind float
  0  0  0  0  0  0  0 |  d = vehic wind non-float
  0  2  0  0 10  0  1 |  e = containers
  1  1  0  0  0  7  0 |  f = tableware
  3  2  0  0  0  1 23 |  g = headlamps
```

在混淆矩阵中，性能好的模型大多数在对角线上，即正确分类的个数。

点击模型，可以编辑模型参数：

<img src="./images/image-20250102110531942.png" alt="image-20250102110531942" style="zoom:50%;" />

J48 参数如下：

- 其中 unpruned 为 False，表示会对决策树进行修剪，将其修改为 True，则不对决策树进行修剪
- 将 `minNumObj` 设置为 15，可以避免特别小的叶节点
- 点击 **More**，可以查看模型相关信息
- 右键可以可视化决策树

<img src="./images/image-20250102110636007.png" alt="image-20250102110636007" style="zoom:50%;" />

可视化决策树：

<img src="./images/image-20250102111519476.png" alt="image-20250102111519476" style="zoom:50%;" />

从 C4.5 到 J48

- ID3 (1979)
- C4.5 (1993)
- C4.8 (1996)
- C5.0 (commercial)

J48 是从 C4.8 的重新实现。

### 1.5 使用 filter

filter 用于数据预处理:

- AllFilter 和 MultiFilter 用于组合多种 filter
- supervised filter 需要 class 属性
- unsupervised filter 不需要 class 属性，使用更为广泛
- filter 又可以分为 attribute filter 和 instance filter

<img src="./images/image-20250102112826528.png" alt="image-20250102112826528" style="zoom:50%;" />

例如，使用 filter 从 weather.nominal.arff 删除一个属性，需要 `Remove` filter，这是一个 unsupservised attribute filter：

- 打开 weather.nominal.arff
- 选择 `Remove` filter
- 设置 filter
- 点击 **Apply**

<img src="./images/image-20250102113106811.png" alt="image-20250102113106811" style="zoom:50%;" />

这里删除了原来的第 3 个属性，humidity。

可以点击 **Undo** 撤销删除操作。

当然，删除属性还有更简单的方法：勾选属性，点击 **Remove**。

也可以删除属性的特定值，例如，删除 humidity 为 `high` 的数据，选择 `RemoveWithValues`，这是一个 instance filter：

<img src="./images/image-20250102113549876.png" alt="image-20250102113549876" style="zoom:50%;" />

选择删除第 3 个属性(humidity)的第 1 个值(high)：

- 设置 attributeIndex 为 3
- nominalIndices 为 1
- 点击 Apply
- 点击 Undo 撤销删除操作

<img src="./images/image-20250102113743304.png" alt="image-20250102113743304" style="zoom:50%;" />

对数据进行合适的预处理，可以得到性能更好的模型：

- 打开 glass.arff
- 运行 J48 (trees>J48)：准确度 66.8224 %
- 删除 **Fe** 特征，再次运行 J48：准确度 67.2897 %
- 删除 RI 和 MG 以外的所有特征，运行 J48：准确度 68.6916 %
- 查看 J48 性能

### 1.6 数据可视化

使用数据可视化面板：

- 打开 iris.arff
- 打开 Visualize 面板

可以看到一个 5x5 的图：

- 点击任意一个子图，可以放大单独显示
- 点击数据点，可以查看数据编号和属性值

<img src="./images/image-20250102123133011.png" alt="image-20250102123133011" style="zoom:50%;" />

点击任意一个子图，都可以放大：

- 点击 X 或 Y 可以设置坐标对应的属性
- 有些数据完整重合，此时可以调整 Jitter 添加随机抖动，从而能够示被覆盖的数据点
- 点击 Rectangle，选择一部分数据，点击 Submit，可以只显示选择区域的点

<img src="./images/image-20250102123657960.png" alt="image-20250102123657960" style="zoom:50%;" />

对训练的模型，右键：Visualize classifier errors，可以查看分类错误图示，对照混淆矩阵查看更清洗。

> [!TIP]
>
> `AddClassification` filter 可以添加模型分类结果。

## 2. 模型评估

### 2.1 分类模型

 如何手动构建决策树：

- 加载 segment-challenge.arff
- 选择 `UserClassifier` 分类器：这是一个手动构建决策树的方法
- 勾选 "Supplied test set"，设置 segment-test.arff 作为单独的测试集
- 点击 Start，弹出如下窗口

<img src="./images/image-20250102142145757.png" alt="image-20250102142145757" style="zoom:50%;" />

- 在 Data Visualizer 查看 region-centroid-row 与 intensity-mean 的视图

<img src="./images/image-20250102142709907.png" alt="image-20250102142709907" style="zoom:50%;" />

- 用 Rectangle 选择一部分数据（红色点），作为一个类别，可以在 Tree Visualizer 查看当前决策树

<img src="./images/image-20250102143342494.png" alt="image-20250102143342494" style="zoom:50%;" />

- 再选择一部分（紫色）作为下一个类别
- 依次下去

在这过程中也可以不使用矩形，而是用多边形来选择。

J48 创建的 tree 通常比自定义的好。

> [!TIP]
>
> 如果没找到 `UserClassifier`，可以在 Package Manager 单独下载。

### 2.2 训练和测试

机器学习的基本流程：

<img src="./images/image-20250102145913185.png" alt="image-20250102145913185" style="zoom: 33%;" />

如果只有一个数据集，可以将其拆分为训练集和测试集。

下面用 J48 分析 segment 数据集：

- 打开 segment-challenge.arff 文件
- 选择 J48 决策树模型
- 设置测试集 segement-test.arff
-  运行，准确度：96.1728 %

如果没有单独的测试集，可以勾选 **Percentage split**，设置训练集和测试集的比例。

### 2.3 重复训练和测试

- 加载 segment-challenge.arff 数据集
- 选择 J48 模型
- 设置 percentage split 为 90%
- 运行：96.7% 准确度
- 在 More options 中设置不同的 seed 重复运行

<img src="./images/image-20250102152615731.png" alt="image-20250102152615731" style="zoom:50%;" />

| Seed     | Accuracy |
| -------- | -------- |
| 1 (默认) | 96.7     |
| 2        | 94       |
| 3        | 94       |
| 4        | 96.7     |
| 5        | 95.3     |
| 6        | 96.7     |
| 7        | 92       |
| 8        | 94       |
| 9        | 93.3     |
| 10       | 94.7     |

然后计算平均值和标准差：
$$
\overline{x}=\frac{\sum x_i}{n}=0.949
$$

$$
\sigma=\sqrt{\frac{\sum (x_i-\overline{x})^2}{n-1}}=0.018
$$

 **总结**：J48 可以得到约 95% 的准确度，偏差在 2% 以内，即 93%-97% 的准确度。

> [!NOTE]
>
> 以上操作的基本假设：训练集和测试集是从无穷总体独立抽样获得。
>
> 得到的结果不可避免有误差，通过设置 seed 和重复实现，可以获得相对可靠的准确度估计值。

### 2.4 基线准确度

- 加载数据集 diabetes.arff
- 勾选 Percentage split 测试选项，取默认的 66%
- 尝试如下分类模型
  - trees -> J48：76%
  - bayes -> NaiveBayes：77%
  - lazy -> IBK：73%
  - rules -> PART：74%

回到 diabetes.arff 数据集，该数据集包含 500 negative，268 positive，如果对所有数据都猜测为 negative，得到精度：500/768=65%。

ZeroR 模型就是该原理：选择 rules->ZeroR，勾选 Use training set，就会得到 65%。

ZeroR 用比例最高的类别作为所有类别，它几乎不用训练，得到的准确度可以作为 baseline。

在少数情况，**baseline 可能是最好的结果**：

- 选择 supermarket.arff 数据集
- 测试不同模型：
  - rules -> ZeroR (Use training set)：64%
  - trees -> J48 (Percentage split 66%)：63%
  - bayes -> NaiveBayes：63%
  - lazy -> IBk：**38%**
  - rules -> PART：63%

baseline 的结果最好，主要原因：属性并不适合作为分类的依据。

> [!TIP]
>
> 总是应该先选择一个 baseline 模型，如 ZeroR。简单的总是最好的，所以先尝试简单的，再考虑复杂的分类模型。
>
> 深入了解数据。

### 2.5 交叉验证

交叉验证是对设置不同 seed、重复运行方法（重复预留法）的改进方法，交叉验证系统重复运行预留法，是机器学习的标准评估方法。分层交叉验证（stratified cross-validation）进一步改进交叉验证。

**重复预留法（repeated holdout）**

将数据集分为 10 份，9 份用于训练，1份用于测试；设置不同 seed，重复 10 次。如下图：

<img src="./images/image-20250102162422652.png" alt="image-20250102162422652" style="zoom: 33%;" />

**10-fold cross-validation**

上面的重复预留法的示例，就是 10 倍交叉验证：重复 10 次，计算平均值。

每个数据都有 1 次用作测试，9 次用作训练。

**分层交叉验证（stratified cross-validation）**

是交叉验证的一个简单变体：在进行数据拆分时，保证每一份每个 class 的比例基本相同。

分层交叉验证可以进一步降低评估误差。

运行完交叉验证，还需要使用整个数据集训练一次，作为**最终的模型**：

<img src="./images/image-20250102163517459.png" alt="image-20250102163517459" style="zoom: 33%;" />

总结：

- 交叉验证比重复预留法更准确
- 分层交叉验证更好，Weka 默认采用分层交叉验证
- 使用 10 倍交叉验证，Weka 需要运行 11 次算法：前面 10 次估计模型性能，最后一次用整个数据集，得到最终模型。
- 建议：
  - 数据足够大：用 percentage split，比如二分类问题，10,000 条数据，10% 用于测试，基本足够了
  - 否则用 10 倍交叉验证

### 2.6 交叉验证结果

为什么交叉验证比重复预留法好，下面通过一个示例演示：

- 加载 diabetes.arff 数据集
- 获取基线准确度（rules -> ZeroR）：65.1%
- trees -> J48
  - 10-fold cross-validation：73.8%

10 倍交叉验证也可以用不同 seed 重复 10 次：

| 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 73.8 | 75.0 | 75.5 | 75.5 | 74.4 | 75.6 | 73.6 | 74.0 | 74.5 | 73.0 |

同样重复执行预留法 10 次，结果如下：

<img src="./images/image-20250102165509501.png" alt="image-20250102165509501" style="zoom: 33%;" />

可以发现，**交叉验证比预留法的标准差要小很多**。

## 3. 简单分类模型

从简单模型开始：

- 一个属性就足够了
- 属性贡献相同，互相独立
- 使用少量属性的决策树
- 只用计算未知样本与训练样本的距离
- 属性的线性组合

没有完美的机器学习算法，机器学习算法的成功取决于具体领域。

### 3.1 从简单模型开始

**OneR**：只用一个属性构建一个单层的决策树。

基础版本：

- 每个值一个分支
- 每个分支分配最高频率类别
- 错误率：分支中不属于多余类别的比例
- 选择错误率最低的属性

以 weather.nominal.arff 数据集为例：

<img src="./images/image-20250102171724676.png" alt="image-20250102171724676" style="zoom: 33%;" />

用 weka 执行 OneR：

- 打开 weather.nominal.arff 数据集
- 选择 ZeroR，作为 baseline：64.2857 %
- 选择 OneR (rules -> OneR)，用 cross-validation 验证：42.8571 %

抛硬币也有 50% 的准确度，所以 OneR 在这个数据集上的效果很差。

由于采用了 10-fold 交叉验证，所以用运行 11 次 OneR，第 11 次用整个数据集训练模型。

### 3.2 过拟合

下面用 OneR 来解释过拟合（overfitting）。对 weather.numeric.arff 数据集，可以对数据集中 14 个样本生成 包含 14 个分叉的 tree：

<img src="./images/image-20250102182150588.png" alt="image-20250102182150588" style="zoom: 50%;" />

这样就可以获得准确率 100% 的模型。

OneR 有限制复杂规则的参数，不过这里为了解释拟合，跳过这一点。

Weka 操作：

- 打开 weather.numeric.arff 文件
- 选择 OneR 模型
- 结果如下

```
outlook:
	sunny	-> no
	overcast	-> yes
	rainy	-> yes
(10/14 instances correct)
```

- 删除 outlook 特征，再次尝试

```
humidity:
	< 82.5	-> yes
	>= 82.5	-> no
(10/14 instances correct)
```

OneR 有一个参数 `minBucketSize`，表示将数值属性离散化的最小 bucket-size，默认为 6，如果将其调整为 1，就会得到一个分叉很多，过拟合的模型：

```
temperature:
	< 64.5	-> yes
	< 66.5	-> no
	< 70.5	-> yes
	< 71.5	-> no
	< 77.5	-> yes
	< 80.5	-> no
	< 84.0	-> yes
	>= 84.0	-> no
(13/14 instances correct)
```

再以 diabetes.arff 数据集为例：

- 打开 diabetes.arff 文件
- 选择 ZeroR，使用 cross-validation：65.1%
- 选择 OneR，使用 cross-validation：71.5%，查看规则

```
=== Classifier model (full training set) ===

plas:
	< 114.5	-> tested_negative
	< 115.5	-> tested_positive
	< 127.5	-> tested_negative
	< 128.5	-> tested_positive
	< 133.5	-> tested_negative
	< 135.5	-> tested_positive
	< 143.5	-> tested_negative
	< 152.5	-> tested_positive
	< 154.5	-> tested_negative
	>= 154.5	-> tested_positive
(587/768 instances correct)
```

将 `minBucketSize` 修改为 1，得到一个特别复杂的模型，准确率低于 baseline：57.1615 %，**典型的过拟合**：

```
=== Classifier model (full training set) ===

pedi:
	< 0.1265	-> tested_negative
	< 0.1275	-> tested_positive
	< 0.1285	-> tested_negative
	< 0.1295	-> tested_positive
	< 0.1345	-> tested_negative
	< 0.1355	-> tested_positive
	< 0.1405	-> tested_negative
	...
	< 1.275	-> tested_negative
	< 1.3969999999999998	-> tested_positive
	< 1.837	-> tested_negative
	< 2.3085	-> tested_positive
	< 2.3745000000000003	-> tested_negative
	>= 2.3745000000000003	-> tested_positive
(672/768 instances correct)
```

如果用 training-set 作为评估，可以发现过拟合的模型效果很好：87.5 %。过拟合模型在训练集上看着很好，但是泛化性能差。

为了避免过拟合，最好的方式是：将数据集拆分为 training, test, validation 三部分。使用 training 和 test 数据集来选择模型，再使用 validation dataset 评估模型。

### 3.3 概率模型

OneR：使用一个属性建模。

相反的策略：使用所有属性，这些属性同样重要。这就是 Naive Bayes 方法。Naive Bayes 方法基于两个假设：

- 所有属性同样重要
- 这些属性在统计上独立的（从一种属性的值无法推测出其它属性的值）

独立假设通常不成立，但是 Naive Bayes 方法在实践中很有效。

贝叶斯理论：
$$
\text{Pr}(H|E)=\frac{\text{Pr}(E|H)\text{Pr}(H)}{\text{Pr}(E)}
$$
其中：

- 在分类模型中，H 代表类别，E 代表样本

- $Pr(H)$ 为 H 的先验概率，即在得到证据前事件的概率，就是 baseline 概率，例如在天气数据中，有 9 个 yes 和 5 个 no，因此 play 是 yes 的概率为 9/14，play 是 no 的概率为 5/14，根据贝叶斯公式，可以基于证据（样本数据）来校正 $Pr(H)$，得到所谓的 H 的后验概率，即知道证据后的概率
- $Pr(H|E)$ 为 H 的后验概率，即得到证据后事件的概率
- Naive 假设：证据（样本）由统计上独立的部分组成，在天气数据集中，指 4 个不同属性值相互独立，因此

$$
Pr(H|E)=\frac{Pr(E_1|H)Pr(E_2|H)\cdots Pr(E_n|H)Pr(H)}{Pr(E)}
$$

以天气数据集为例：

<img src="./images/image-20250102192537620.png" alt="image-20250102192537620" style="zoom:33%;" />



### 3.4 决策树

### 3.5 修剪决策树

### 3.6 最近邻



## 4. 更多分类模型

## 5. 汇总

## 参考

- https://www.bilibili.com/video/BV1Hb411q7Bf

