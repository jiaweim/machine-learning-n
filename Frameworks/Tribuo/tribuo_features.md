# Trubuo 特点

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