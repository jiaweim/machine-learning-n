# 逻辑回归

- [逻辑回归](#逻辑回归)
  - [简介](#简介)
  - [Sklearn 示例](#sklearn-示例)
  - [参考](#参考)

***

## 简介

逻辑回归实现为 `LogisticRegression`。虽然它叫逻辑回归，其实是个线性分类模型。在该模型中，使用 logistic 函数计算单次试验的概率。logistic 函数即 sigmoid 函数，如下：

![](images/2023-01-13-09-15-42.png)

如果用逻辑回归进行二分类，通常先确定一个概率阈值，超出该概率就认为是类别 1，低于该阈值就认为是类别 0。例如，将阈值设为 0.5：

![](images/2023-01-13-09-31-33.png)

此时，概率 y=0.8 对应类别 1，概率 y=0.3 对应类别 0.

该实现适合 binary, One-vs-Rest 或 multinomial 逻辑回归，提供可选的 $l_1$, $l_2$ 和 Elastic-Net 正则化。

> **NOTE: 正则化**
> 默认应用正则化，这在机器学习中很常见，但在统计学中不常见。正则化的一个好处是提高了数值稳定性。没有正则化等价于将 C 值设置很高。

> NOTE: 逻辑回归

## Sklearn 示例

1. 导入包

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

2. 加载数据

这里使用 kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 数据集，该数据集包含信用卡交易信息，包含欺诈和非欺诈两类数据。

```python
credit_card = pd.read_csv('datasets/creditcard.csv')
```

3. 可视化

查看欺诈和非欺诈样本数：

```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x="Class", data=credit_card)
plt.title("# Fraud vs. NonFraud")
plt.xlabel("Class (1==Fraud)")
```

![](images/logit1.png)

280k 样本，只有 400 多个是 Fraud 样本。使用这类极度不平衡的数据集建立模型，模型很可能会始终预测结果为 NonFraud，虽然精度很高，但是毫无用处。

查看数据集中 NonFraud 样本比例：

```python
base_line_accuracy = 1 - np.sum(credit_card.Class) / credit_card.shape[0]
base_line_accuracy
```

```txt
0.9982725143693799
```

此时 accuracy 对评估模型没有帮助，因此使用 AUC ROC 来评估模型质量。

4. 拆分数据集

首先将 `Class` 从数据集中分离出来，然后拆分为训练集和测试集：

```python
X = credit_card.drop(columns='Class', axis=1)
y = credit_card.Class.values

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

5. 构建模型和训练

在构建模型前，使用标准 scaler 函数对特征进行标准化，然后创建 `LogisticRegression` 模型。

这里使用 `LogisticRegression` 的默认参数：

- **penalty**: 默认为 `L2`
- **C**: 默认 1.0，正则化强度的导数
- **solver**: 默认 'lbfgs'，优化算法

下面使用 `Pipeline` 创建模型，以简化模型构建：

```python
scaler = StandardScaler()
lr = LogisticRegression()
model = Pipeline([('standardize', scaler),
                  ('log_reg', lr)])
```

训练模型：

```python
model.fit(X_train, y_train)
```

```txt
Pipeline(steps=[('standardize', StandardScaler()),
                ('log_reg', LogisticRegression())])
```

6. 训练打分

下面创建 confusion matrix 来评估模型。另外，用 `roc_auc_score()` 函数获取 AUC。

我们也计算了精度，虽然对不平衡的数据集精度用处不大。

```python
y_train_hat = model.predict(X_train)
y_train_hat_probs = model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_hat) * 100
train_auc_roc = roc_auc_score(y_train, y_train_hat_probs) * 100

print('Confusion matrix:\n', confusion_matrix(y_train, y_train_hat))
print("Training AUC: %.4f %%" % train_auc_roc)
print("Training accuracy: %.4f %%" % train_accuracy)
```

```txt
Confusion matrix:
 [[213198     28]
 [   135    244]]
Training AUC: 98.0170 %
Training accuracy: 99.9237 %
```

7. 测试打分

对测试集计算混淆矩阵、AUC ROC 和精度。

结果表明，上面训练的模型对 113 个欺诈交易检测出 68个，召回率 60%。另外有 12 个假阳性结果，考虑到数据集规模，这个值已经很小了。

```python
y_test_hat = model.predict(X_test)
y_test_hat_probs = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_hat) * 100
test_auc_roc = roc_auc_score(y_test, y_test_hat_probs) * 100

print("Confusion matrix:\n", confusion_matrix(y_test, y_test_hat))
print("Testing AUC: %.4f %%" % test_auc_roc)
print("Testing accuracy: %.4f %%" % test_accuracy)
```

```txt
Confusion matrix:
 [[71077    12]
 [   45    68]]
Testing AUC: 97.4516 %
Testing accuracy: 99.9199 %
```

为了更清晰的展示结果，用 `classification_report()` 查看模型在测试集上的精度和召回率。

从第一行的 f1-score 可以发现模型能够识别 70% 的欺诈案例：

```python
print(classification_report(y_test, y_test_hat, digits=6))
```

```txt
              precision    recall  f1-score   support

           0   0.999367  0.999831  0.999599     71089
           1   0.850000  0.601770  0.704663       113

    accuracy                       0.999199     71202
   macro avg   0.924684  0.800801  0.852131     71202
weighted avg   0.999130  0.999199  0.999131     71202
```

我们可以通过更改 C 值或选择其它 solver 来改进模型性能。

也可以使用 sklearn 的 imblearn 模块中的SMOTE 算法来平衡数据集以提高性能。

## 参考

- https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- https://machinelearningknowledge.ai/python-sklearn-logistic-regression-tutorial-with-example/
- https://developers.google.com/machine-learning/crash-course/logistic-regression/video-lecture
