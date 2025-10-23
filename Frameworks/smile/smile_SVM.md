# SVM 示例

## libsvm 示例数据

1. **默认参数训练**

```java
System.out.println("svmguide1");
MathEx.setSeed(19650218);

SparseDataset<Integer> train = Read.libsvm("G:\\tools\\libsvm\\svmguide1");
SparseDataset<Integer> test = Read.libsvm("G:\\tools\\libsvm\\svmguide1.t");

int n = train.size();
double[][] x = new double[n][4];
int[] y = new int[n];
for (int i = 0; i < n; i++) {
    SampleInstance<SparseArray, Integer> sample = train.get(i);
    for (SparseArray.Entry e : sample.x()) {
        x[i][e.index()] = e.value();
    }
    y[i] = sample.y() > 0 ? +1 : -1;
}

n = test.size();
double[][] testX = new double[n][4];
int[] testy = new int[n];
for (int i = 0; i < n; i++) {
    SampleInstance<SparseArray, Integer> sample = test.get(i);
    for (SparseArray.Entry entry : sample.x()) {
        testX[i][entry.index()] = entry.value();
    }
    testy[i] = sample.y() > 0 ? +1 : -1;
}

GaussianKernel kernel = new GaussianKernel(0.25);
SVM<double[]> model = SVM.fit(x, y, kernel, new SVM.Options(1));

int[] prediction = model.predict(testX);
int error = Error.of(testy, prediction);
System.out.format("Test Error = %d, Accuracy = %.2f%%%n", error, 100.0 - 100.0 * error / testX.length);
```

```
svmguide1
14:17:27 [main] INFO smile.base.svm.LASVM[165] - 1000 iterations, 1006 support vectors
14:17:27 [main] INFO smile.base.svm.LASVM[165] - 2000 iterations, 2004 support vectors
14:17:27 [main] INFO smile.base.svm.LASVM[165] - 3000 iterations, 3000 support vectors
14:17:27 [main] INFO smile.base.svm.LASVM[467] - Finalizing the training by reprocess.
14:17:27 [main] INFO smile.base.svm.LASVM[470] - 1000 reprocess iterations.
14:17:27 [main] INFO smile.base.svm.LASVM[470] - 2000 reprocess iterations.
14:17:27 [main] INFO smile.base.svm.LASVM[457] - 3089 samples, 3089 support vectors, 978 bounded
Test Error = 1994, Accuracy = 50.15%
```

说明：

- `GaussianKernel` 又称为 RBF kernel，与 libsvm 的默认kernel 相同
- libsvm 的 C 默认为 1，gamma 默认 1/num_features，为了与 libsvm 的参数一致，将 C 设为 1，gamas 设为 1/4

2. **对数据进行缩放**

```java
MathEx.setSeed(19650218);

SparseDataset<Integer> train = Read.libsvm("G:\\tools\\libsvm\\svmguide1");
SparseDataset<Integer> test = Read.libsvm("G:\\tools\\libsvm\\svmguide1.t");

int n = train.size();
double[][] x = new double[n][4];
int[] y = new int[n];
for (int i = 0; i < n; i++) {
    SampleInstance<SparseArray, Integer> sample = train.get(i);
    for (SparseArray.Entry e : sample.x()) {
        x[i][e.index()] = e.value();
    }
    y[i] = sample.y() > 0 ? +1 : -1;
}

DataFrame trainFrame = DataFrame.of(x, "1", "2", "3", "4");
InvertibleColumnTransform transform = Scaler.fit(trainFrame, "1", "2", "3", "4");
DataFrame trainScale = transform.apply(trainFrame);

n = test.size();
double[][] testX = new double[n][4];
int[] testy = new int[n];
for (int i = 0; i < n; i++) {
    SampleInstance<SparseArray, Integer> sample = test.get(i);
    for (SparseArray.Entry entry : sample.x()) {
        testX[i][entry.index()] = entry.value();
    }
    testy[i] = sample.y() > 0 ? +1 : -1;
}
DataFrame testFrame = DataFrame.of(testX, "1", "2", "3", "4");
DataFrame testScale = transform.apply(testFrame);

double[][] trainScaleX = trainScale.toArray();

GaussianKernel kernel = new GaussianKernel(0.25);
SVM<double[]> model = SVM.fit(trainScaleX, y, kernel, new SVM.Options(1));

int[] prediction = model.predict(testScale.toArray());
int error = Error.of(testy, prediction);
System.out.format("Test Error = %d, Accuracy = %.2f%%%n", error, 100.0 - 100.0 * error / testX.length);
```

```
14:32:30 [main] INFO smile.base.svm.LASVM[165] - 1000 iterations, 171 support vectors
14:32:30 [main] INFO smile.base.svm.LASVM[165] - 2000 iterations, 289 support vectors
14:32:30 [main] INFO smile.base.svm.LASVM[165] - 3000 iterations, 400 support vectors
14:32:30 [main] INFO smile.base.svm.LASVM[467] - Finalizing the training by reprocess.
14:32:30 [main] INFO smile.base.svm.LASVM[457] - 3089 samples, 406 support vectors, 385 bounded
Test Error = 131, Accuracy = 96.73%
```

说明：

- 使用 `Scaler` 将每个 feature 缩放到 0-1 之间
- 特征缩放之后，准确性提高许多

**3. 参数选择**

`GaussianKernel` 的参数 `sigma` 以及 SVM 的 `C` 都可以优化。

- libsvm 默认采用 5-fold 交叉验证
- C 范围：log2c 范围 [-5,15], step=2
- g 范围：log2g 范围 [-15,3], step=2

> [!WARNING]
>
> smile 中所有 classifiers 要求分类标签为 [0,k]，即不支持负数标签。`CrossValidation` 同样要求 labels 不是负数。
>
> 但是，LASVM 要求标签为 `{+1,-1}`，换言之，cross-validation 目前不能用于 LASVM。
