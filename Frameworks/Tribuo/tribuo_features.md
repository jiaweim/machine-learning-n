# Trubuo 特点

## Provenance

Tribuo 的 Model, Dataset 和 Evaluation 都包含来源信息，它们知道创建模型泗洪的参数、数据变换和文件。根据这些信息就可以从头开始构建模型。

## 类型安全

Tribuo 是强类型（同 Java）。每个模型都知道它输出和输入类型。

## 互操性

Tribuo 提供了与 XGBoost 和 Tensorflow 等流行机器学习库的结构，并支持 ONNX 模型交换格式。

ONNX 通过 onnx-runtime 实现，可用于部署用其它软件包和其它语言构建的模型（如 scikit-learn）。许多 Tribuo 模型可以导出为 ONNX 格式，以便部署在其它系统或云服务中。

## 算法

Tribuo 支持许多流行的机器学习算法。这些算法按类型分组，Tribuo 的抽象接口使得切换实现很容易。

### General predictors

Tribuo 提供多种可用于各种预测任务的算法。

| Algorithm       | Implementation | Notes                                                        |
| --------------- | -------------- | ------------------------------------------------------------ |
| Bagging         | Tribuo         | Can use any Tribuo trainer as the base learner               |
| Random Forest   | Tribuo         | Can use any Tribuo tree trainer as the base learner          |
| Extra Trees     | Tribuo         | For both classification and regression                       |
| K-NN            | Tribuo         | Includes options for several parallel backends, as well as a single threaded backend |
| Neural Networks | TensorFlow     | Train a neural network in TensorFlow via the Tribuo wrapper. Models can be deployed using the ONNX interface or the TF interface |

### Classification

| Algorithm                       | Implementation      | Notes                                                        |
| ------------------------------- | ------------------- | ------------------------------------------------------------ |
| Linear models                   | Tribuo              | Uses SGD and allows any gradient optimizer                   |
| Factorization Machines          | Tribuo              | Uses SGD and allows any gradient optimizer                   |
| CART                            | Tribuo              |                                                              |
| SVM-SGD                         | Tribuo              | An implementation of the Pegasos algorithm                   |
| Adaboost.SAMME                  | Tribuo              | Can use any Tribuo classification trainer as the base learner |
| Multinomial Naive Bayes         | Tribuo              |                                                              |
| Regularised Linear Models       | LibLinear           |                                                              |
| SVM                             | LibSVM or LibLinear | LibLinear only supports linear SVMs                          |
| Gradient Boosted Decision Trees | XGBoost             |                                                              |

### Regression

| Algorithm                       | Implementation      | Notes                                      |
| ------------------------------- | ------------------- | ------------------------------------------ |
| Linear models                   | Tribuo              | Uses SGD and allows any gradient optimizer |
| Factorization Machines          | Tribuo              | Uses SGD and allows any gradient optimizer |
| CART                            | Tribuo              |                                            |
| Lasso                           | Tribuo              | Using the LARS algorithm                   |
| Elastic Net                     | Tribuo              | Using the co-ordinate descent algorithm    |
| Regularised Linear Models       | LibLinear           |                                            |
| SVM                             | LibSVM or LibLinear | LibLinear only supports linear SVMs        |
| Gradient Boosted Decision Trees | XGBoost             |                                            |

### Clustering

| Algorithm | Implementation | Notes                                                        |
| --------- | -------------- | ------------------------------------------------------------ |
| HDBSCAN*  | Tribuo         |                                                              |
| K-Means   | Tribuo         | Includes both sequential and parallel backends, and the K-Means++ initialisation algorithm |



## 参考

- https://tribuo.org/learn/4.3/docs/features.html