# LIBSVM API

2025-05-13
@author Jiawei Mao

***

## 简介

LIBSVM 函数和数据结构都声明在头文件 `svm.h` 中，所以需要在源码中声明 `#include svm.h`。

在 `svm-train.c` 和 `svm-predict.c` 可以看到如何使用这些函数。在 `svm.h` 中用 `LIBSVM_VERSION` 定义了 SVM 版本。

在分类测试数据前，需要使用训练数据构建 SVM 模型，将模型保存在文件中。然后就可以使用保存的 SVM 模型分类数据。

## 训练

用于训练的函数：
```cpp
struct svm_model *svm_train(const struct svm_problem *prob,
					const struct svm_parameter *param);
```
该函数根据训练数据和参数创建 SVM 模型。

`svm_problem` 包含训练数据集：

```cpp
struct svm_problem
{
    int l;
    double *y;
    struct svm_node **x;
};
```
其中 ：
- `l` 表示训练数据大小
- `y` 是目标值数组（对分类为整数值，对回归为实数）
- `x` 数组指针，每个指向一个 `svm_node` 数组，对应训练向量

例如，对如下数据：
|LABEL|ATTR1|ATTR2|ATTR3|ATTR4|ATTR5|
|---|---|---|---|---|---|
|1  | 0  | 0.1|0.2| 0  |0|
|2  | 0  | 0.1|0.3|-1.2|0|
|1  | 0.4| 0  |0  | 0  |0|
|2  | 0  | 0.1|0  | 1.4|0.5|
|3  |-0.1|-0.2|0.1| 1.1|0.1|

对应的 `svm_problem` 是：
```
l = 5
y -> 1 2 1 2 3
x ->[ ] -> (2,0.1) (3,0.2) (-1,?)
    [ ] -> (2,0.1) (3,0.3) (4,-1.2) (-1,?)
    [ ] -> (1,0.4) (-1,?)
    [ ] -> (2,0.1) (4,1.4) (5,0.5) (-1,?)
    [ ] -> (1,-0.1) (2,-0.2) (3,0.1) (4,1.1) (5,0.1) (-1,?)
```
其中 `x` 的0 值省略，每个值都带有索引，对应`(index, value)` 保存在 `svm_node` 中：
```cpp
struct svm_node
{
    int index;
    double value;
};
```
`index = -1` 表示向量结尾。值索引必须按升序排列。

## svm_parameter

struct `svm_parameter` 描述 SVM 模型参数：
```cpp
struct svm_parameter
{
    int svm_type;
    int kernel_type;
    int degree;	/* for poly */
    double gamma;	/* for poly/rbf/sigmoid */
    double coef0;	/* for poly/sigmoid */

    /* these are for training only */
    double cache_size; /* in MB */
    double eps;	/* stopping criteria */
    double C;	/* for C_SVC, EPSILON_SVR, and NU_SVR */
    int nr_weight;		/* for C_SVC */
    int *weight_label;	/* for C_SVC */
    double* weight;		/* for C_SVC */
    double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;	/* for EPSILON_SVR */
    int shrinking;	/* use the shrinking heuristics */
    int probability; /* do probability estimates */
};
```
### svm_type

svm_type 支持的类型：

|svm_type|说明|
|---|----|
|C_SVC|C-SVM classification, default|
|NU_SVC|nu-SVM classification|
|ONE_CLASS|one-class-SVM|
|EPSILON_SVR|epsilon-SVM regression|
|NU_SVR|nu-SVM regression|

### kernel_type
|kernel_type|说明|
|---|---|
|LINEAR|u'*v|
|POLY|$(\gamma*u'*v + coef8)^{degree}$|
|RBF|$exp(-\gamma*\vert u-v \vert^2)$, default|
|SIGMOID|$tanh(\gamma*u'*v+coef0)$|
|PRECOMPUTER|kernel values in training_set_file|

### `degree`
核函数的 degree，默认3.

`cache_size` 是缓存大小，单位为 MB。

`C` 是违反约束的成本（cost of contraints violation）。

`eps` is the stopping criterion. (一般 nu-SVC 中使用 0.00001，其它使用 0.001).

`nu` 是 nv-SVM, nu-SVR 和 one-class-SVM 中的参数。

`p` is the epsilon in epsilon-insensitive loss function of epsilon-SVM regression. 

`shrinking = 1` means shrinking is conducted; = 0 otherwise.

`probability = 1` means model with probability information is obtained; = 0 otherwise.

`nr_weight`, `weight_label`, and `weight` are used to change the penalty for some classes (If the weight for a class is not changed, it is set to 1). This is useful for training classifier using unbalanced input data or with asymmetric misclassification cost.

`nr_weight` is the number of elements in the array `weight_label` and `weight`. Each `weight[i]` corresponds to `weight_label[i]`, meaning that the penalty of class `weight_label[i]` is scaled by a factor of `weight[i]`.

 If you do not want to change penalty for any of the classes, just set `nr_weight` to 0.

> [!NOTE]
>
> 由于 svm_model 包含指向 svm_problem 的指针，因此如果仍在使用 `svm_train()` 生成的 `svm_model`， 则不能释放 `svm_problem` 使用的内存。 


## Java 版

预编译的 java 类放在 `libsvm.jar` 中，源码在 java 目录。使用方法：
```sh
java -classpath libsvm.jar svm_train <arguments>
java -classpath libsvm.jar svm_predict <arguments>
java -classpath libsvm.jar svm_toy
java -classpath libsvm.jar svm_scale <arguments>
```

库的使用类似于 C 版本，可用的函数如下：
```java
public class svm {
	public static final int LIBSVM_VERSION=336;
	public static svm_model svm_train(svm_problem prob, svm_parameter param);
	public static void svm_cross_validation(svm_problem prob, svm_parameter param, int nr_fold, double[] target);
	public static int svm_get_svm_type(svm_model model);
	public static int svm_get_nr_class(svm_model model);
	public static void svm_get_labels(svm_model model, int[] label);
	public static void svm_get_sv_indices(svm_model model, int[] indices);
	public static int svm_get_nr_sv(svm_model model);
	public static double svm_get_svr_probability(svm_model model);
	public static double svm_predict_values(svm_model model, svm_node[] x, double[] dec_values);
	public static double svm_predict(svm_model model, svm_node[] x);
	public static double svm_predict_probability(svm_model model, svm_node[] x, double[] prob_estimates);
	public static void svm_save_model(String model_file_name, svm_model model) throws IOException
	public static svm_model svm_load_model(String model_file_name) throws IOException
	public static String svm_check_parameter(svm_problem prob, svm_parameter param);
	public static int svm_check_probability_model(svm_model model);
	public static void svm_set_print_string_function(svm_print_interface print_func);
}
```

### svm_train

在 Java 版本中， svm_node[] 不是以 index=-1 的节点结尾。

可以通过如下方式制定输出格式：
```java
your_print_func = new svm_print_interface()
{
    public void print(String s)
    {
        // your own format
    }
};
svm.svm_set_print_string_function(your_print_func);
```

### svm_model

`l` 是支持向量的数据量。`SV` 和 `sv_coef` 分别是支持向量和对应的系数。假设有 $k$ 个类别。对类别为 `j` 的样本，对应的 `sv_coef` 包含 $(k-1)y*alpha$ 个向量，其中 alpha 是以下两类问题的解：

- 1 vs j, 2 vs j, ..., j-1 vs j, j vs j+1, j vs j+2,,,., j vs k 

例如，如果有 4 个类别，则 `sv_coef` 和 `SV` 的样式：

```
+-+-+-+--------------------+
|1|1|1|                    |
|v|v|v|  SVs from class 1  |
|2|3|4|                    |
+-+-+-+--------------------+
|1|2|2|                    |
|v|v|v|  SVs from class 2  |
|2|3|4|                    |
+-+-+-+--------------------+
|1|2|3|                    |
|v|v|v|  SVs from class 3  |
|3|3|4|                    |
+-+-+-+--------------------+
|1|2|3|                    |
|v|v|v|  SVs from class 4  |
|4|4|4|                    |
+-+-+-+--------------------+
```

