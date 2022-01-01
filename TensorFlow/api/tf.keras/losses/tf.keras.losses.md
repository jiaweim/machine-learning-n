# tf.keras.losses

- [tf.keras.losses](#tfkeraslosses)
  - [类](#类)
  - [函数](#函数)
  - [参考](#参考)

2022-01-01, 13:58
***

## 类

|类|说明|
|---|---|
|BinaryCrossentropy|Computes the cross-entropy loss between true labels and predicted labels.|
|CategoricalCrossentropy|Computes the crossentropy loss between the labels and predictions.|
|CategoricalHinge|Computes the categorical hinge loss between y_true and y_pred.|
|CosineSimilarity|Computes the cosine similarity between labels and predictions.|
|Hinge|Computes the hinge loss between y_true and y_pred.|
|Huber|Computes the Huber loss between y_true and y_pred.|
|KLDivergence|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|LogCosh|Computes the logarithm of the hyperbolic cosine of the prediction error.|
|Loss|Loss base class.|
|MeanAbsoluteError|Computes the mean of absolute difference between labels and predictions.|
|MeanAbsolutePercentageError|Computes the mean absolute percentage error between y_true and y_pred.|
|MeanSquaredError|Computes the mean of squares of errors between labels and predictions.|
|MeanSquaredLogarithmicError|Computes the mean squared logarithmic error between y_true and y_pred.|
|Poisson|Computes the Poisson loss between y_true and y_pred.|
|Reduction|Types of loss reduction.|
|SparseCategoricalCrossentropy|Computes the crossentropy loss between the labels and predictions.|
|SquaredHinge|Computes the squared hinge loss between y_true and y_pred.|

## 函数

|函数|说明|
|---|---|
|KLD(...)|Computes Kullback-Leibler divergence loss between y_true and y_pred.|
|MAE(...)|Computes the mean absolute error between labels and predictions.|
|MAPE(...)|Computes the mean absolute percentage error between y_true and y_pred.|
|MSE(...)|Computes the mean squared error between labels and predictions.|
|MSLE(...)|Computes the mean squared logarithmic error between y_true and y_pred.|
|binary_crossentropy(...)|Computes the binary crossentropy loss.|
|categorical_crossentropy(...)|Computes the categorical crossentropy loss.|
|categorical_hinge(...)|Computes the categorical hinge loss between y_true and y_pred.|
|cosine_similarity(...)|Computes the cosine similarity between labels and predictions.|
|deserialize(...)|Deserializes a serialized loss class/function instance.|
|get(...)|Retrieves a Keras loss as a function/Loss class instance.|
|hinge(...)|Computes the hinge loss between y_true and y_pred.|
|huber(...)|Computes Huber loss value.|
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
|serialize(...)|Serializes loss function or Loss instance.|
|sparse_categorical_crossentropy(...)|Computes the sparse categorical crossentropy loss.|
|squared_hinge(...)|Computes the squared hinge loss between y_true and y_pred.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/losses
