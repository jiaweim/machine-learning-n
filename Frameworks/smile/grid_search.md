# 参数选择

smile 提供的 `smile.hpo.Hyperparameters` 类实现了与 grid-search 相同的超参数选择功能。

超参数（hyperparamter）是在算法学习之前设置的参数，其它的参数则是算法通过训练得出。超参数可以分为：

- model hyperparamters：将算法对训练集拟合之前无法确定
- algorithm hyperparamters：原则上对模型性能没有影响，但会影响学习的速度和质量

例如，神经网络的拓扑结构和大小属于模型超参数，而学习率和 mini-batch size 为算法超参数。

示例：调整随机森林的超参数：

```java
import smile.io.*;
import smile.data.formula.Formula;
import smile.validation.*;
import smile.classification.RandomForest;

var hp = new Hyperparameters()
    .add("smile.random.forest.trees", 100) // a fixed value
    .add("smile.random.forest.mtry", new int[] {2, 3, 4}) // an array of values to choose
    .add("smile.random.forest.max.nodes", 100, 500, 50); // range [100, 500] with step 50

var train = Read.arff("data/weka/segment-challenge.arff");
var test = Read.arff("data/weka/segment-test.arff");
var formula = Formula.lhs("class");
var testy = formula.y(test).toIntArray();

hp.grid().forEach(prop -&gt; {
    var model = RandomForest.fit(formula, train, prop);
    var pred = model.predict(test);
    System.out.println(prop);
    System.out.format("Accuracy = %.2f%%%n", (100.0 * Accuracy.of(testy, pred)));
    System.out.println(ConfusionMatrix.of(testy, pred));
});
```

`add` 用于添加参数，前面为参数名称。

- 添加参数，只有一个参数值，表示参数固定

```java
Hyperparameters add(String name, int value);
Hyperparameters add(String name, double value);
...
```

- 多个参数值，表示从这些参数中选择

```java
Hyperparameters add(String name, int[] values);
Hyperparameters add(String name, double[] values);
Hyperparameters add(String name, String[] values);
Hyperparameters add(String name, double start, double end, double step);
```

- grid

```java
Stream<Properties> grid()
```

`grid` 生成参数组合 `Property` 的 `stream`。每个 `Properties` 包含一组参数组合。

> [!NOTE]
>
> Hyperparameters 本质上只是辅助生成不同参数组合。