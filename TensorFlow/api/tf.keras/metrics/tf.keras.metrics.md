# tf.keras.metrics

- [tf.keras.metrics](#tfkerasmetrics)
  - [类](#类)
  - [函数](#函数)
  - [参考](#参考)

2022-01-01, 14:20
***

## 类

|类|说明|
|---|---|
|AUC|Approximates the AUC (Area under the curve) of the ROC or PR curves.|
|Accuracy|Calculates how often predictions equal labels.|
|BinaryAccuracy|Calculates how often predictions match binary labels.|
|BinaryCrossentropy|Computes the crossentropy metric between the labels and predictions.|
|CategoricalAccuracy|Calculates how often predictions match one-hot labels.|
|CategoricalCrossentropy|Computes the crossentropy metric between the labels and predictions.|
|CategoricalHinge|Computes the categorical hinge metric between y_true and y_pred.|
|CosineSimilarity|Computes the cosine similarity between the labels and predictions.|
|FalseNegatives|Calculates the number of false negatives.|
|FalsePositives|Calculates the number of false positives.|
|Hinge|Computes the hinge metric between y_true and y_pred.|
|KLDivergence|Computes Kullback-Leibler divergence metric between y_true and y_pred.|
|LogCoshError|Computes the logarithm of the hyperbolic cosine of the prediction error.|
|Mean|Computes the (weighted) mean of the given values.|
|MeanAbsoluteError|Computes the mean absolute error between the labels and predictions.|
|MeanAbsolutePercentageError|Computes the mean absolute percentage error between y_true and y_pred.|
|MeanIoU|Computes the mean Intersection-Over-Union metric.|
|MeanMetricWrapper|Wraps a stateless metric function with the Mean metric.|
|MeanRelativeError|Computes the mean relative error by normalizing with the given values.|
|MeanSquaredError|Computes the mean squared error between y_true and y_pred.|
|MeanSquaredLogarithmicError|Computes the mean squared logarithmic error between y_true and y_pred.|
|MeanTensor|Computes the element-wise (weighted) mean of the given tensors.|
|Metric|Encapsulates metric logic and state.|
|Poisson|Computes the Poisson metric between y_true and y_pred.|
|Precision|Computes the precision of the predictions with respect to the labels.|
|PrecisionAtRecall|Computes best precision where recall is >= specified value.|
|Recall|Computes the recall of the predictions with respect to the labels.|
|RecallAtPrecision|Computes best recall where precision is >= specified value.|
|RootMeanSquaredError|Computes root mean squared error metric between y_true and y_pred.|
|SensitivityAtSpecificity|Computes best sensitivity where specificity is >= specified value.|
|SparseCategoricalAccuracy|Calculates how often predictions match integer labels.|
|SparseCategoricalCrossentropy|Computes the crossentropy metric between the labels and predictions.|
|SparseTopKCategoricalAccuracy|Computes how often integer targets are in the top K predictions.|
|SpecificityAtSensitivity|Computes best specificity where sensitivity is >= specified value.|
|SquaredHinge|Computes the squared hinge metric between y_true and y_pred.|
|Sum|Computes the (weighted) sum of the given values.|
|TopKCategoricalAccuracy|Computes how often targets are in the top K predictions.|
|TrueNegatives|Calculates the number of true negatives.|
|TruePositives|Calculates the number of true positives.|

## 函数

|函数|说明|
|---|---|
|KLD(...)|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|MAE(...)|Computes the mean absolute error between labels and predictions.|
|MAPE(...)|Computes the mean absolute percentage error between y_true and y_pred.|
|MSE(...)|Computes the mean squared error between labels and predictions.|
|MSLE(...)|Computes the mean squared logarithmic error between y_true and y_pred.|
|binary_accuracy(...)|Calculates how often predictions match binary labels.|
|binary_crossentropy(...)|Computes the binary crossentropy loss.|
|categorical_accuracy(...)|Calculates how often predictions match one-hot labels.|
|categorical_crossentropy(...)|Computes the categorical crossentropy loss.|
|deserialize(...)|Deserializes a serialized metric class/function instance.|
|get(...)|Retrieves a Keras metric as a function/Metric class instance.|
|hinge(...)|Computes the hinge loss between y_true and y_pred.|
|kl_divergence(...)|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|kld(...)|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|kullback_leibler_divergence(...)|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|log_cosh(...)|Logarithm of the hyperbolic cosine of the prediction error.|
|logcosh(...)|Logarithm of the hyperbolic cosine of the prediction error.|
|mae(...)|Computes the mean absolute error between labels and predictions.|
|mape(...)|Computes the mean absolute percentage error between y_true and y_pred.|
|mean_absolute_error(...)|Computes the mean absolute error between labels and predictions.|
|mean_absolute_percentage_error(...)|Computes the mean absolute percentage error between y_true and y_pred.|
|mean_squared_error(...)|Computes the mean squared error between labels and predictions.|
|mean_squared_logarithmic_error(...)|Computes the mean squared logarithmic error between y_true and y_pred.|
|mse(...)|Computes the mean squared error between labels and predictions.|
|msle(...)|Computes the mean squared logarithmic error between y_true and y_pred.|
|poisson(...)|Computes the Poisson loss between y_true and y_pred.|
|serialize(...)|Serializes metric function or Metric instance.|
|sparse_categorical_accuracy(...)|Calculates how often predictions match integer labels.|
|sparse_categorical_crossentropy(...)|Computes the sparse categorical crossentropy loss.|
|sparse_top_k_categorical_accuracy(...)|Computes how often integer targets are in the top K predictions.|
|squared_hinge(...)|Computes the squared hinge loss between y_true and y_pred.|
|top_k_categorical_accuracy(...)|Computes how often targets are in the top K predictions.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/metrics
