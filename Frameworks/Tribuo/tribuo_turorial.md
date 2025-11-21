# Tribuo 教程

## 简介

如果刚接触 Tribuo，可以按照[分类](#分类)教程学习基本操作。它介绍了训练和测试模型的方方面面，以及如何加载和保存模型。

> [!NOTE]
>
> 所有 Tribuo 教程都以 Jupyter 笔记本形式提供。虽然 Jupyter 原生不支持 Java，但可以通过 IJava kernel 添加 Java 支持。在教程中使用 Java 关键字 `var`，因此需要 Java 10+，其中 model card 和 reproducibility 教程需要 Java 17+。Tribuo 本身支持 Java 8+。

Tribuo  被拆分为许多模块，允许你只加载其中的一部分。这对限制 jar 的大小和范围很有用，还可以排除包含 native 依赖项的部分。

不过在学习 Tribuo 的时候，使用 `tribuo-all` 最方便，它包含 Tribuo 的所有组件以及第三方依赖项。

**Maven**

```xml
<dependency>
    <groupId>org.tribuo</groupId>
    <artifactId>tribuo-all</artifactId>
    <version>4.3.2</version>
    <type>pom</type>
</dependency>
```

## 分类

下面展示如何使用 Tribuo 的分类模型，使用 Fisher 的著名 Iris 数据集，预测鸢尾花的物种。这个使用简单的 logistic-regression，并研究 Tribuo 在每个模型中存储的信息。

### Setup

首先，需要下载 iris 数据集：

```sh
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
```

然后记在 Trubuo jars。这里需要使用分类 jar，以及 json interop jar 来读取和写入信息 。

```java
%jars ./olcut-core-5.1.4-SNAPSHOT.jar
%jars ./tribuo-classification-experiments-4.0.0-SNAPSHOT-jar-with-dependencies.jar
%jars ./tribuo-json-4.0.0-SNAPSHOT-jar-with-dependencies.jar
```

```java
import java.nio.file.Paths;
```

导入 org.tribuo 包的所有内容、CSV loader，以及分类包。

```java
import org.tribuo.*;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
```

还有针对来源系统的导入：

```java
import com.fasterxml.jackson.databind.*;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.config.json.*;
```

### 加载数据

在 Tribuo 中，所有预测类型都有一个关联的 `OutputFactory` 实现，它可以根据输入创建合适的 `Output` 子类。这里使用 `LabelFactory`，因为执行的是多类分类。将 `labelFactory` 传递给 `CSVLoader`，`CSVLoader` 会读取所有列的数据到 `DataSource`。

```java
var labelFactory = new LabelFactory();
var csvLoader = new CSVLoader<>(labelFactory);
```

前面下载的 iris 数据没有标题，这里手动创建标题，并将标题、文件路径以及输出变量信息（这里为 "species"）提供给 load 方法。Irises 没有预定义 train/test 拆分，因此我们创建一个，将 70% 的数据用于训练。

```java
var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
var irisesSource = csvLoader.loadDataSource(Paths.get("bezdekIris.data"),"species",irisHeaders);
var irisSplitter = new TrainTestSplitter<>(irisesSource,0.7,1L);
```

将训练集和测试集分别输入到各自的 `Dataset`。这些 `Dataset` 会计算所有必要的元数据，如 feature domain 和 output domain。对训练集，最好使用 `MutableDataset`，因为它可以应用转换操作。准备好数据集，就可以训练模型啦：

```java
var trainingDataset = new MutableDataset<>(irisSplitter.getTrain());
var testingDataset = new MutableDataset<>(irisSplitter.getTest());
System.out.println(String.format("Training data size = %d, number of features = %d, number of classes = %d",trainingDataset.size(),trainingDataset.getFeatureMap().size(),trainingDataset.getOutputInfo().size()));
System.out.println(String.format("Testing data size = %d, number of features = %d, number of classes = %d",testingDataset.size(),testingDataset.getFeatureMap().size(),testingDataset.getOutputInfo().size()));
```

```
Training data size = 105, number of features = 4, number of classes = 3
Testing data size = 45, number of features = 4, number of classes = 3
```

### 训练模型

现在实例化训练器，看看它的默认超参数有哪些。可以使用 `LinearSGDTrainer` 对其参数进行完全控制。

```java
Trainer<Label> trainer = new LogisticRegressionTrainer();
System.out.println(trainer.toString());
```

```
LinearSGDTrainer(objective=LogMulticlass,optimiser=AdaGrad(initialLearningRate=1.0,epsilon=0.1,initialValue=0.0),epochs=5,minibatchSize=1,seed=12345)
```

这是一个使用 logistic-loss 的线性模型，使用 `AdaGrad` 训练 5 个周期。

现在训练模型。与其它软件包一样，当你准备好训练算法和训练数据后，训练就非常简单。

```java
Model<Label> irisModel = trainer.train(trainingDataset);
```

### 评估模型

训练好模型后，需要评估它的质量。为此，我们可以从 `labelFactory` 获取合适的 `Evaluator`（或者直接实例化），然后向 evaluator 传递模型和测试数据集。也可以提供 data-source，替代 dataset。`LabelEvaluator` 类实现了所有常见的分类指标，每个指标都可以单独检查。`LabelEvaluation.toString()` 生成格式良好的指标摘要。

```java
var evaluator = new LabelEvaluator();
var evaluation = evaluator.evaluate(irisModel,testingDataset);
System.out.println(evaluation.toString());
```

```
Class                           n          tp          fn          fp      recall        prec          f1
Iris-versicolor                16          16           0           1       1.000       0.941       0.970
Iris-virginica                 15          14           1           0       0.933       1.000       0.966
Iris-setosa                    14          14           0           0       1.000       1.000       1.000
Total                          45          44           1           1
Accuracy                                                                    0.978
Micro Average                                                               0.978       0.978       0.978
Macro Average                                                               0.978       0.980       0.978
Balanced Error Rate                                                         0.022
```

[Precision, recall, and F1](https://en.wikipedia.org/wiki/Precision_and_recall) 是评估多分类模型时使用的标准指标。

我们还可以打印混淆矩阵：

```java
System.out.println(evaluation.getConfusionMatrix().toString());
```

```
                   Iris-versicolor   Iris-virginica      Iris-setosa
Iris-versicolor                 16                0                0
Iris-virginica                   1               14                0
Iris-setosa                      0                0               14
```

### 模型元数据

Tribuo 跟踪模型的特征和输出域。这意味着可以在不访问原始训练数据的情况下运行类似 LIME 的技术，并且可以添加检查以确保特定输入是否在训练模型所见的范围内。

下面看看 Irises 模型的特征域：

```java
var featureMap = irisModel.getFeatureIDMap();
for (var v : featureMap) {
    System.out.println(v.toString());
    System.out.println();
}
```

```
CategoricalFeature(name=petalLength,id=0,count=105,map={1.2=1, 6.9=1, 3.6=1, 3.0=1, 1.7=4, 4.9=4, 4.4=3, 3.5=2, 5.9=2, 5.4=1, 4.0=4, 1.4=12, 4.5=4, 5.0=2, 5.5=3, 6.7=2, 3.7=1, 1.9=1, 6.0=2, 5.2=1, 5.7=2, 4.2=2, 4.7=2, 4.8=4, 1.6=4, 5.8=2, 3.8=1, 6.3=1, 3.3=1, 1.0=1, 5.6=4, 5.1=5, 4.6=3, 4.1=2, 1.5=9, 1.3=4, 3.9=3, 6.6=1, 6.1=2})

CategoricalFeature(name=petalWidth,id=1,count=105,map={2.0=3, 0.5=1, 1.2=3, 0.3=6, 1.6=2, 0.1=3, 0.4=5, 2.5=3, 2.3=4, 1.7=2, 1.1=3, 2.1=4, 0.6=1, 1.4=6, 1.0=5, 2.4=1, 1.8=12, 0.2=20, 1.9=4, 1.5=7, 1.3=8, 2.2=2})

CategoricalFeature(name=sepalLength,id=2,count=105,map={6.9=3, 6.4=3, 7.4=1, 4.9=4, 4.4=1, 5.9=3, 5.4=5, 7.2=3, 7.7=3, 5.0=8, 6.2=2, 5.5=5, 6.7=7, 6.0=3, 5.2=2, 6.5=3, 5.7=4, 4.7=2, 4.8=3, 5.8=4, 5.3=1, 6.8=3, 6.3=5, 7.3=1, 5.6=6, 5.1=7, 4.6=4, 7.6=1, 7.1=1, 6.6=2, 6.1=5})

CategoricalFeature(name=sepalWidth,id=3,count=105,map={2.0=1, 2.8=10, 3.6=4, 2.3=3, 2.5=5, 3.1=8, 3.8=4, 3.0=19, 2.6=4, 4.4=1, 3.3=4, 3.5=4, 2.4=2, 3.2=10, 2.9=5, 3.7=3, 3.4=6, 2.2=2, 3.9=2, 4.2=1, 2.7=7})
```

我们可以看到 4 个特征，以及它们值的直方图。该信息可用于从每个特征进行采样，为 LIME 等本地解释器构建候选样本，也可以用于检查特征值的范围。特征信息在训练时冻结，因此当特征集稀疏时，也可用于检查训练集中某个特征出现的次数。

### 模型来源

现代应用部署许多不同类型的机器学习模型，为应用的不同方面提供帮助。然而，大多数机器学习包不支持跟踪和重建模型。在 Tribuo 中，每个模型都会跟踪其来源。它知道它是如何创建的、何时创建的，以及涉及哪些数据。下面看看鸢尾花模型的数据来源（data provenance）。Tribuo 默认会在每个来源对象的 `toString()` 中以人类可读的格式打印来源，而且所有信息都可以通过编程方式访问。

```java
var provenance = irisModel.getProvenance();
System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getDatasetProvenance().getSourceProvenance()));
```

```
TrainTestSplitter(
	class-name = org.tribuo.evaluation.TrainTestSplitter
	source = CSVDataSource(
			class-name = org.tribuo.data.csv.CSVDataSource
			headers = List[
				sepalLength
				sepalWidth
				petalLength
				petalWidth
				species
			]
			rowProcessor = RowProcessor(...)
			quote = "
			outputRequired = true
			outputFactory = LabelFactory(
					class-name = org.tribuo.classification.LabelFactory
				)
			separator = ,
			dataPath = D:\data\tribuo\bezdekIris.data
			resource-hash = 0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC
			file-modified-time = 2025-11-21T17:58:51.852+08:00
			datasource-creation-time = 2025-11-21T19:20:17.518255400+08:00
			host-short-name = DataSource
		)
	train-proportion = 0.7
	seed = 1
	size = 150
	is-train = true
)
```

我们可以看到模型是在一个 datasource 上训练的，datasource 被拆分为两部分，采用特定的随机 seed 和拆分比例。原始数据源是一个 CSV 文件，还记录了文件修改时间和 SHA-256 hash。

我们可以以类似方法检查训练器的来源，以了解训练算法。

```java
System.out.println(ProvenanceUtil.formattedProvenanceString(provenance.getTrainerProvenance()));
```

```
LogisticRegressionTrainer(
	class-name = org.tribuo.classification.sgd.linear.LogisticRegressionTrainer
	seed = 12345
	minibatchSize = 1
	shuffle = true
	epochs = 5
	optimiser = AdaGrad(
			class-name = org.tribuo.math.optimisers.AdaGrad
			epsilon = 0.1
			initialLearningRate = 1.0
			initialValue = 0.0
			host-short-name = StochasticGradientOptimiser
		)
	loggingInterval = 1000
	objective = LogMulticlass(
			class-name = org.tribuo.classification.sgd.objectives.LogMulticlass
			host-short-name = LabelObjective
		)
	tribuo-version = 4.3.2
	train-invocation-count = 0
	is-sequence = false
	host-short-name = Trainer
)
```

正如预期，这里使用 `LogisticRegressionTrainer` 训练的模型，该训练器使用 `AdaGrad` 作为梯度下降算法。

可以从模型中提取来源信息并保存为 json 文件，保存为单独的副本：

```java
ObjectMapper objMapper = new ObjectMapper();
objMapper.registerModule(new JsonProvenanceModule());
objMapper = objMapper.enable(SerializationFeature.INDENT_OUTPUT);
```

json 来源信息很冗长，但提供了另一种人类可读的序列化格式。

```java
String jsonProvenance = objMapper.writeValueAsString(ProvenanceUtil.marshalProvenance(provenance));
System.out.println(jsonProvenance);
```

```json
[ {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "linearsgdmodel-0",
  "object-class-name" : "org.tribuo.classification.sgd.linear.LinearSGDModel",
  "provenance-class" : "org.tribuo.provenance.ModelProvenance",
  "map" : {
    "instance-values" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.MapMarshalledProvenance",
      "map" : { }
    },
    "tribuo-version" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "tribuo-version",
      "value" : "4.3.2",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "java-version" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "java-version",
      "value" : "25.0.1",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "trainer" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "trainer",
      "value" : "logisticregressiontrainer-2",
      "provenance-class" : "org.tribuo.provenance.impl.TrainerProvenanceImpl",
      "additional" : "",
      "is-reference" : true
    },
    "os-arch" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "os-arch",
      "value" : "amd64",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "trained-at" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "trained-at",
      "value" : "2025-11-21T19:31:14.393497400+08:00",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "os-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "os-name",
      "value" : "Windows 11",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "dataset" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "dataset",
      "value" : "mutabledataset-1",
      "provenance-class" : "org.tribuo.provenance.DatasetProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.classification.sgd.linear.LinearSGDModel",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "mutabledataset-1",
  "object-class-name" : "org.tribuo.MutableDataset",
  "provenance-class" : "org.tribuo.provenance.DatasetProvenance",
  "map" : {
    "num-features" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "num-features",
      "value" : "4",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "num-examples" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "num-examples",
      "value" : "105",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "num-outputs" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "num-outputs",
      "value" : "3",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "tribuo-version" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "tribuo-version",
      "value" : "4.3.2",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "datasource" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "datasource",
      "value" : "traintestsplitter-3",
      "provenance-class" : "org.tribuo.evaluation.TrainTestSplitter$SplitDataSourceProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "transformations" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ ]
    },
    "is-sequence" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "is-sequence",
      "value" : "false",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "is-dense" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "is-dense",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.MutableDataset",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "logisticregressiontrainer-2",
  "object-class-name" : "org.tribuo.classification.sgd.linear.LogisticRegressionTrainer",
  "provenance-class" : "org.tribuo.provenance.impl.TrainerProvenanceImpl",
  "map" : {
    "seed" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "seed",
      "value" : "12345",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "tribuo-version" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "tribuo-version",
      "value" : "4.3.2",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "minibatchSize" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "minibatchSize",
      "value" : "1",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "train-invocation-count" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "train-invocation-count",
      "value" : "0",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "is-sequence" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "is-sequence",
      "value" : "false",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "shuffle" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "shuffle",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "epochs" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "epochs",
      "value" : "5",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "optimiser" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "optimiser",
      "value" : "adagrad-4",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
      "additional" : "",
      "is-reference" : true
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "Trainer",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.classification.sgd.linear.LogisticRegressionTrainer",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "loggingInterval" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "loggingInterval",
      "value" : "1000",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "objective" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "objective",
      "value" : "logmulticlass-5",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
      "additional" : "",
      "is-reference" : true
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "traintestsplitter-3",
  "object-class-name" : "org.tribuo.evaluation.TrainTestSplitter",
  "provenance-class" : "org.tribuo.evaluation.TrainTestSplitter$SplitDataSourceProvenance",
  "map" : {
    "train-proportion" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "train-proportion",
      "value" : "0.7",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "seed" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "seed",
      "value" : "1",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "size" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "size",
      "value" : "150",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "source" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "source",
      "value" : "csvdatasource-6",
      "provenance-class" : "org.tribuo.data.csv.CSVDataSource$CSVDataSourceProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.evaluation.TrainTestSplitter",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "is-train" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "is-train",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "adagrad-4",
  "object-class-name" : "org.tribuo.math.optimisers.AdaGrad",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "epsilon" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "epsilon",
      "value" : "0.1",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "initialLearningRate" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "initialLearningRate",
      "value" : "1.0",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "initialValue" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "initialValue",
      "value" : "0.0",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "StochasticGradientOptimiser",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.math.optimisers.AdaGrad",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "logmulticlass-5",
  "object-class-name" : "org.tribuo.classification.sgd.objectives.LogMulticlass",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "LabelObjective",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.classification.sgd.objectives.LogMulticlass",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "csvdatasource-6",
  "object-class-name" : "org.tribuo.data.csv.CSVDataSource",
  "provenance-class" : "org.tribuo.data.csv.CSVDataSource$CSVDataSourceProvenance",
  "map" : {
    "resource-hash" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "resource-hash",
      "value" : "0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance",
      "additional" : "SHA256",
      "is-reference" : false
    },
    "headers" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "headers",
        "value" : "sepalLength",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "headers",
        "value" : "sepalWidth",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "headers",
        "value" : "petalLength",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "headers",
        "value" : "petalWidth",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "headers",
        "value" : "species",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      } ]
    },
    "rowProcessor" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "rowProcessor",
      "value" : "rowprocessor-7",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
      "additional" : "",
      "is-reference" : true
    },
    "file-modified-time" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "file-modified-time",
      "value" : "2025-11-21T17:58:51.852+08:00",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "quote" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "quote",
      "value" : "\"",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.CharProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "outputRequired" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "outputRequired",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "datasource-creation-time" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "datasource-creation-time",
      "value" : "2025-11-21T19:31:14.343577700+08:00",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "outputFactory" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "outputFactory",
      "value" : "labelfactory-15",
      "provenance-class" : "org.tribuo.classification.LabelFactory$LabelFactoryProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "separator" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "separator",
      "value" : ",",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.CharProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "DataSource",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.csv.CSVDataSource",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "dataPath" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "dataPath",
      "value" : "D:\\data\\tribuo\\bezdekIris.data",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "rowprocessor-7",
  "object-class-name" : "org.tribuo.data.columnar.RowProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "metadataExtractors" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ ]
    },
    "fieldProcessorList" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "fieldProcessorList",
        "value" : "doublefieldprocessor-9",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
        "additional" : "",
        "is-reference" : true
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "fieldProcessorList",
        "value" : "doublefieldprocessor-10",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
        "additional" : "",
        "is-reference" : true
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "fieldProcessorList",
        "value" : "doublefieldprocessor-11",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
        "additional" : "",
        "is-reference" : true
      }, {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "fieldProcessorList",
        "value" : "doublefieldprocessor-12",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
        "additional" : "",
        "is-reference" : true
      } ]
    },
    "featureProcessors" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ ]
    },
    "responseProcessor" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "responseProcessor",
      "value" : "fieldresponseprocessor-13",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
      "additional" : "",
      "is-reference" : true
    },
    "weightExtractor" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "weightExtractor",
      "value" : "fieldextractor-14",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.NullConfiguredProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "replaceNewlinesWithSpaces" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "replaceNewlinesWithSpaces",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "regexMappingProcessors" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.MapMarshalledProvenance",
      "map" : { }
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "RowProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.RowProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "labelfactory-15",
  "object-class-name" : "org.tribuo.classification.LabelFactory",
  "provenance-class" : "org.tribuo.classification.LabelFactory$LabelFactoryProvenance",
  "map" : {
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.classification.LabelFactory",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "doublefieldprocessor-9",
  "object-class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "fieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "fieldName",
      "value" : "petalLength",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "onlyFieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "onlyFieldName",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "throwOnInvalid" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "throwOnInvalid",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "FieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "doublefieldprocessor-10",
  "object-class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "fieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "fieldName",
      "value" : "petalWidth",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "onlyFieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "onlyFieldName",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "throwOnInvalid" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "throwOnInvalid",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "FieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "doublefieldprocessor-11",
  "object-class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "fieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "fieldName",
      "value" : "sepalWidth",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "onlyFieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "onlyFieldName",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "throwOnInvalid" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "throwOnInvalid",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "FieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "doublefieldprocessor-12",
  "object-class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "fieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "fieldName",
      "value" : "sepalLength",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "onlyFieldName" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "onlyFieldName",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "throwOnInvalid" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "throwOnInvalid",
      "value" : "true",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "FieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "fieldresponseprocessor-13",
  "object-class-name" : "org.tribuo.data.columnar.processors.response.FieldResponseProcessor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl",
  "map" : {
    "uppercase" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "uppercase",
      "value" : "false",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "fieldNames" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "fieldNames",
        "value" : "species",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      } ]
    },
    "defaultValues" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ListMarshalledProvenance",
      "list" : [ {
        "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
        "key" : "defaultValues",
        "value" : "",
        "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
        "additional" : "",
        "is-reference" : false
      } ]
    },
    "displayField" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "displayField",
      "value" : "false",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "outputFactory" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "outputFactory",
      "value" : "labelfactory-15",
      "provenance-class" : "org.tribuo.classification.LabelFactory$LabelFactoryProvenance",
      "additional" : "",
      "is-reference" : true
    },
    "host-short-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "host-short-name",
      "value" : "ResponseProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    },
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.processors.response.FieldResponseProcessor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
}, {
  "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance",
  "object-name" : "fieldextractor-14",
  "object-class-name" : "org.tribuo.data.columnar.FieldExtractor",
  "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.impl.NullConfiguredProvenance",
  "map" : {
    "class-name" : {
      "marshalled-class" : "com.oracle.labs.mlrg.olcut.provenance.io.SimpleMarshalledProvenance",
      "key" : "class-name",
      "value" : "org.tribuo.data.columnar.FieldExtractor",
      "provenance-class" : "com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance",
      "additional" : "",
      "is-reference" : false
    }
  }
} ]
```

另外，`Model.toString()`  中也包含模型来源，不过该格式不适合机器读取。

```java
System.out.println(irisModel.toString());
```

```
inear-sgd-model - Model(class-name=org.tribuo.classification.sgd.linear.LinearSGDModel,dataset=Dataset(class-name=org.tribuo.MutableDataset,datasource=SplitDataSourceProvenance(className=org.tribuo.evaluation.TrainTestSplitter,innerSourceProvenance=DataSource(class-name=org.tribuo.data.csv.CSVDataSource,headers=[sepalLength, sepalWidth, petalLength, petalWidth, species],rowProcessor=RowProcessor(class-name=org.tribuo.data.columnar.RowProcessor,metadataExtractors=[],fieldProcessorList=[FieldProcessor(class-name=org.tribuo.data.columnar.processors.field.DoubleFieldProcessor,fieldName=petalLength,onlyFieldName=true,throwOnInvalid=true,host-short-name=FieldProcessor), FieldProcessor(class-name=org.tribuo.data.columnar.processors.field.DoubleFieldProcessor,fieldName=petalWidth,onlyFieldName=true,throwOnInvalid=true,host-short-name=FieldProcessor), FieldProcessor(class-name=org.tribuo.data.columnar.processors.field.DoubleFieldProcessor,fieldName=sepalWidth,onlyFieldName=true,throwOnInvalid=true,host-short-name=FieldProcessor), FieldProcessor(class-name=org.tribuo.data.columnar.processors.field.DoubleFieldProcessor,fieldName=sepalLength,onlyFieldName=true,throwOnInvalid=true,host-short-name=FieldProcessor)],featureProcessors=[],responseProcessor=ResponseProcessor(class-name=org.tribuo.data.columnar.processors.response.FieldResponseProcessor,uppercase=false,fieldNames=[species],defaultValues=[],displayField=false,outputFactory=OutputFactory(class-name=org.tribuo.classification.LabelFactory),host-short-name=ResponseProcessor),weightExtractor=null,replaceNewlinesWithSpaces=true,regexMappingProcessors={},host-short-name=RowProcessor),quote=",outputRequired=true,outputFactory=OutputFactory(class-name=org.tribuo.classification.LabelFactory),separator=,,dataPath=D:\data\tribuo\bezdekIris.data,resource-hash=SHA-256[0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC],file-modified-time=2025-11-21T17:58:51.852+08:00,datasource-creation-time=2025-11-21T19:33:33.101386400+08:00,host-short-name=DataSource),trainProportion=0.7,seed=1,size=150,isTrain=true),transformations=[],is-sequence=false,is-dense=true,num-examples=105,num-features=4,num-outputs=3,tribuo-version=4.3.2),trainer=Trainer(class-name=org.tribuo.classification.sgd.linear.LogisticRegressionTrainer,seed=12345,minibatchSize=1,shuffle=true,epochs=5,optimiser=StochasticGradientOptimiser(class-name=org.tribuo.math.optimisers.AdaGrad,epsilon=0.1,initialLearningRate=1.0,initialValue=0.0,host-short-name=StochasticGradientOptimiser),loggingInterval=1000,objective=LabelObjective(class-name=org.tribuo.classification.sgd.objectives.LogMulticlass,host-short-name=LabelObjective),tribuo-version=4.3.2,train-invocation-count=0,is-sequence=false,host-short-name=Trainer),trained-at=2025-11-21T19:33:33.147630600+08:00,instance-values={},tribuo-version=4.3.2,java-version=25.0.1,os-name=Windows 11,os-arch=amd64)
```

Evaluations 还会记录模型来源和测试数据来源。下面使用 JSON 格式的另一种形式，更易于 阅读，但不太精确。这种形式适合参考，但不能用于重建原始来源对象，因为它将所有内容都转换为字符串。

```java
String jsonEvaluationProvenance = objMapper.writeValueAsString(ProvenanceUtil.convertToMap(evaluation.getProvenance()));
System.out.println(jsonEvaluationProvenance);
```

```
{
  "tribuo-version" : "4.3.2",
  "dataset-provenance" : {
    "num-features" : "4",
    "num-examples" : "45",
    "num-outputs" : "3",
    "tribuo-version" : "4.3.2",
    "datasource" : {
      "train-proportion" : "0.7",
      "seed" : "1",
      "size" : "150",
      "source" : {
        "resource-hash" : "0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC",
        "headers" : [ "sepalLength", "sepalWidth", "petalLength", "petalWidth", "species" ],
        "rowProcessor" : {
          "metadataExtractors" : [ ],
          "fieldProcessorList" : [ {
            "fieldName" : "petalLength",
            "onlyFieldName" : "true",
            "throwOnInvalid" : "true",
            "host-short-name" : "FieldProcessor",
            "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
          }, {
            "fieldName" : "petalWidth",
            "onlyFieldName" : "true",
            "throwOnInvalid" : "true",
            "host-short-name" : "FieldProcessor",
            "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
          }, {
            "fieldName" : "sepalWidth",
            "onlyFieldName" : "true",
            "throwOnInvalid" : "true",
            "host-short-name" : "FieldProcessor",
            "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
          }, {
            "fieldName" : "sepalLength",
            "onlyFieldName" : "true",
            "throwOnInvalid" : "true",
            "host-short-name" : "FieldProcessor",
            "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
          } ],
          "featureProcessors" : [ ],
          "responseProcessor" : {
            "uppercase" : "false",
            "fieldNames" : [ "species" ],
            "defaultValues" : [ "" ],
            "displayField" : "false",
            "outputFactory" : {
              "class-name" : "org.tribuo.classification.LabelFactory"
            },
            "host-short-name" : "ResponseProcessor",
            "class-name" : "org.tribuo.data.columnar.processors.response.FieldResponseProcessor"
          },
          "weightExtractor" : {
            "class-name" : "org.tribuo.data.columnar.FieldExtractor"
          },
          "replaceNewlinesWithSpaces" : "true",
          "regexMappingProcessors" : { },
          "host-short-name" : "RowProcessor",
          "class-name" : "org.tribuo.data.columnar.RowProcessor"
        },
        "file-modified-time" : "2025-11-21T17:58:51.852+08:00",
        "quote" : "\"",
        "outputRequired" : "true",
        "datasource-creation-time" : "2025-11-21T19:35:56.396323+08:00",
        "outputFactory" : {
          "class-name" : "org.tribuo.classification.LabelFactory"
        },
        "separator" : ",",
        "host-short-name" : "DataSource",
        "class-name" : "org.tribuo.data.csv.CSVDataSource",
        "dataPath" : "D:\\data\\tribuo\\bezdekIris.data"
      },
      "class-name" : "org.tribuo.evaluation.TrainTestSplitter",
      "is-train" : "false"
    },
    "transformations" : [ ],
    "is-sequence" : "false",
    "is-dense" : "true",
    "class-name" : "org.tribuo.MutableDataset"
  },
  "class-name" : "org.tribuo.provenance.EvaluationProvenance",
  "model-provenance" : {
    "instance-values" : { },
    "tribuo-version" : "4.3.2",
    "java-version" : "25.0.1",
    "trainer" : {
      "seed" : "12345",
      "tribuo-version" : "4.3.2",
      "minibatchSize" : "1",
      "train-invocation-count" : "0",
      "is-sequence" : "false",
      "shuffle" : "true",
      "epochs" : "5",
      "optimiser" : {
        "epsilon" : "0.1",
        "initialLearningRate" : "1.0",
        "initialValue" : "0.0",
        "host-short-name" : "StochasticGradientOptimiser",
        "class-name" : "org.tribuo.math.optimisers.AdaGrad"
      },
      "host-short-name" : "Trainer",
      "class-name" : "org.tribuo.classification.sgd.linear.LogisticRegressionTrainer",
      "loggingInterval" : "1000",
      "objective" : {
        "host-short-name" : "LabelObjective",
        "class-name" : "org.tribuo.classification.sgd.objectives.LogMulticlass"
      }
    },
    "os-arch" : "amd64",
    "trained-at" : "2025-11-21T19:35:56.447514+08:00",
    "os-name" : "Windows 11",
    "dataset" : {
      "num-features" : "4",
      "num-examples" : "105",
      "num-outputs" : "3",
      "tribuo-version" : "4.3.2",
      "datasource" : {
        "train-proportion" : "0.7",
        "seed" : "1",
        "size" : "150",
        "source" : {
          "resource-hash" : "0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC",
          "headers" : [ "sepalLength", "sepalWidth", "petalLength", "petalWidth", "species" ],
          "rowProcessor" : {
            "metadataExtractors" : [ ],
            "fieldProcessorList" : [ {
              "fieldName" : "petalLength",
              "onlyFieldName" : "true",
              "throwOnInvalid" : "true",
              "host-short-name" : "FieldProcessor",
              "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
            }, {
              "fieldName" : "petalWidth",
              "onlyFieldName" : "true",
              "throwOnInvalid" : "true",
              "host-short-name" : "FieldProcessor",
              "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
            }, {
              "fieldName" : "sepalWidth",
              "onlyFieldName" : "true",
              "throwOnInvalid" : "true",
              "host-short-name" : "FieldProcessor",
              "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
            }, {
              "fieldName" : "sepalLength",
              "onlyFieldName" : "true",
              "throwOnInvalid" : "true",
              "host-short-name" : "FieldProcessor",
              "class-name" : "org.tribuo.data.columnar.processors.field.DoubleFieldProcessor"
            } ],
            "featureProcessors" : [ ],
            "responseProcessor" : {
              "uppercase" : "false",
              "fieldNames" : [ "species" ],
              "defaultValues" : [ "" ],
              "displayField" : "false",
              "outputFactory" : {
                "class-name" : "org.tribuo.classification.LabelFactory"
              },
              "host-short-name" : "ResponseProcessor",
              "class-name" : "org.tribuo.data.columnar.processors.response.FieldResponseProcessor"
            },
            "weightExtractor" : {
              "class-name" : "org.tribuo.data.columnar.FieldExtractor"
            },
            "replaceNewlinesWithSpaces" : "true",
            "regexMappingProcessors" : { },
            "host-short-name" : "RowProcessor",
            "class-name" : "org.tribuo.data.columnar.RowProcessor"
          },
          "file-modified-time" : "2025-11-21T17:58:51.852+08:00",
          "quote" : "\"",
          "outputRequired" : "true",
          "datasource-creation-time" : "2025-11-21T19:35:56.396323+08:00",
          "outputFactory" : {
            "class-name" : "org.tribuo.classification.LabelFactory"
          },
          "separator" : ",",
          "host-short-name" : "DataSource",
          "class-name" : "org.tribuo.data.csv.CSVDataSource",
          "dataPath" : "D:\\data\\tribuo\\bezdekIris.data"
        },
        "class-name" : "org.tribuo.evaluation.TrainTestSplitter",
        "is-train" : "true"
      },
      "transformations" : [ ],
      "is-sequence" : "false",
      "is-dense" : "true",
      "class-name" : "org.tribuo.MutableDataset"
    },
    "class-name" : "org.tribuo.classification.sgd.linear.LinearSGDModel"
  }
}
```

可以看到该来源信息包含模型来源的所有字段，测试数据，拆分比例以及来源的 CSV。

这些来源信息对跟踪模型很有用，结合 config 系统就成为重建模型和实验的强大方法，计划可以完美复制任何 ML 模型。

### 总结

研究 Tribuo 的 CSV 加载机制，如何训练简单分类模型，如何使用测试集评估模型，以及 Tribuo 模型和评估对象中存储了哪些元数据和来源信息。

## 参考

- https://tribuo.org/learn/4.3/tutorials/
