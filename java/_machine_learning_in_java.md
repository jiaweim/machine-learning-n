# Java 机器学习库

## 选择

- 如果必须在原生 Java/JVM 环境运行，推荐使用 Deeplearning4J
- 如果需要处理大型数据集，需要分布式计算能力：推荐 Apache Spark MLlib
- 如果是初学者，需要快速进行传统的分类、聚类分析，或者需要一个带 GUI 的工具：推荐 Weka
- 如果想在 Java 中部署其它框架（如 TensorFlow, PyTorch, Scikit-learn）训练的模型：推荐 ONNX Runtime 或 TensorFlow/PyTorch 的 Java API （通常用户模型部署，而非训练）

对机器学习框架：

- 对新手或学生，专注于传统算法研究，推荐 Weka
- 如果是软件工程师，需要将高性能 ML 集成到 Java/Scale 生产代码，推荐 SMILE
- 如果是在企业环境，需要结构清晰、支持可解释性，易于部署和维护模型的库，推荐 Tribuo

## 简介

- [Tribuo](https://github.com/oracle/tribuo)
- [Weka](https://waikato.github.io/weka-site/index.html)
- [Deep Java Library](https://djl.ai/)
- [Smile](https://haifengl.github.io/)
- [Spark](https://spark.apache.org/mllib/)
- [ELKI](https://elki-project.github.io/)：侧重关系数据库中数据的异常值检测
- [Apache Mahout](https://mahout.apache.org/)：分布式线性代数框架，辅助数学家、统计学家和数据科学家快速实现新的算法。Apache Spark 是对应的开箱即用的分布式后端。
- [JSAT](https://github.com/EdwardRaff/JSAT)：纯 Java 实现的机器学习库。
- [javaml](https://github.com/AbeelLab/javaml)

## 深度学习

[Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)：基于 JVM 的深度学习框架，与 Hadoop 兼容，提供受限玻尔兹曼机、深度信念网络等算法。

[Encog](https://github.com/jeffheaton/encog-java-core)：早期用于神经网络技术的 Java 框架，提供了 SVM、经典神经网络、遗传编程、贝叶斯网络、HMM和遗传算法。可以被 DeepLearning4J 取代，

##  性能比较

https://ceur-ws.org/Vol-2874/paper16.pdf

Weka > 

**速度**

小型数据集，Tribuo 和 SMILE 还可以；大型数据集，则远比 Weka 慢。

**内存**

Weka < Tribuo < SMILE

## 贝叶斯

- [UnBBayes](https://unbbayes.sourceforge.net/index.html)
- [OpenMarkov](https://www.openmarkov.org/users.html)