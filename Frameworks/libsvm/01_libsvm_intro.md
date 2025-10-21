# LIBSVM 概述

2025-05-13⭐
@author Jiawei Mao
***

## 简介

LIBSVM 是一个 SVM 集成工具包，支持：

- 分类：C-SVC, nu-SVC
- 回归：epsilon-SVR, nu-SVR
- 分布估计（one-class SVM）
- 多类别分类

从 2.8开始，libsvm 实现了 SMO-type 算法。LIBSVM 的主要特征：

- 多种 SVM 公式
- 高效的 multi-class 分类
- 用于模型选择的交叉验证
- 概率估计
- 多种 kernels（包括内置的 kernel matrix）
- 用于不平衡数据的加权 SVM
- 包含 C++ 和 Java 源码
- 演示 SVM 分类和回归的 GUI
- 支持 Python, R, MATLAB, Perl, Ruby, Weka, Common LISP, CLISP, Haskell, OCaml, LabVIEW 和 PHP 接口，提供 C# .NET 代码和 CUDA 扩展，一些数据挖掘平台，如 RapidMiner, PCP 和 LIONsolver 也包含
- 一些流行的软件包管理器（如 pip 和 vcpkg）可以轻松地安装 LIBSVM
- 自动模型选择可以生成交叉验证准确率的轮廓图

## Quick Start
如果你是 SVM 新手，并且数据集不大，可以在安装后使用 "tools" 文件夹下的 easy.py 脚本。该脚本自动执行所有任务，从数据缩放到参数选择。

使用方式：
```sh
easy.py training_file [testing_file]
```

## 安装和数据格式
### 安装
在 Unix 系统，使用 `make` 命令构建 `svm-train`, `svm-predict` 和 `svm-scale` 程序。不带参数运行可以查看使用方式。

在其它系统，参考 "Makefile" 构建，或者使用预构建的二进制版本，其中 Windows 的可执行文件在 “windows” 文件夹。

### 数据格式
训练和测试数据集文件格式为：
```
<label> <index1>:<value1> <index2>:<value2> ...
.
.
.
```

示例：
```
1 1:4.530499e+01 2:2.619430e+02 3:-2.311574e-01 4:1.553381e+02
1 1:6.451801e+01 2:1.884440e+02 3:7.265563e-02 4:1.333321e+02
1 1:8.675299e+01 2:3.088610e+02 3:-9.522417e-02 4:1.430497e+02
1 1:5.171198e+01 2:2.807610e+02 3:-1.852275e-01 4:1.526079e+02
0 1:1.785300e+01 2:1.493100e+01 3:1.706039e-01 4:6.352117e+01
0 1:1.681499e+01 2:2.620200e+01 3:1.487285e-01 4:4.935408e+01
0 1:1.794760e+01 2:3.439160e+01 3:6.074293e-01 4:1.535747e+02
0 1:1.643700e+01 2:2.080002e-01 3:4.028665e-01 4:3.551385e+01
```
每行为一个样本数据，并以 '\n' 结尾。样本可以没有特征值（一行全是 0），但是 `<label>` 列不能为空。

对**训练集**中的 `<label>`，有如下几种情况：

- 分类：`<label>` 为整数值，表示其分类，支持多类（multi-class）
- 回归：`<label>` 为目标值，可以是任意实数
- 对 one-class SVM，不使用 `<label>` 值，因此可以是任何数

对**测试集**，`<label>` 仅用于计算准确率。如果未知，可以填充任意数。对 one-class SVM，如果non-outliers/outliers 已知，则在测试文件中的 labels 值必须为 +1/-1，用于评估性能。`<label>` 值使用 C 标准库的 `strtod()` 读取。

`label` 后面的成对值 `<index>:<value>` 为 feature（attribute）：
- `<index>` 是从 1 开始的整数；对预计算（precomputed）内核，`<index>` 从0开始
- `index` 必须升序排列
- `<value>` 实数

下面是一个分类数据样例 "heart_scale"的前两行:
```
+1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1 
-1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1 
```
可以使用 `tools/checkdata.py` 检查数据格式是否正确。

使用样例数据：
- 输入 `svm-train heart_scale`，该程序会读入训练数据，输出模型文件 "heart_scale.model"；
- 如果你有测试数据 heart_scale.t，可以输入 `svm-predict heart_scale.t heart_scale.model output` 查看预测准确性。输出文件 `output` 包含预测的分类。

对分类，如果训练集只有一个分类（即所有 labels 一样），`svm-train` 会输出警告信息 `Warning: training data in only one class. See README for details`，表示训练集不平衡。在测试时会直接返回训练集中的 label。

在该包中还有其它一些工具：

**svm-scale**  

用于缩放输入数据。

**svm-toy**  

这是一个简单的图形界面，用于展示 SVM 如何在平面中分隔数据。你可以在界面中点击绘制数据点。

- "change" 按钮选择数据点分类 1, 2 或 3（最多支持三类）
- "load" 按钮从文件载入数据
- "save" 按钮保存数据到文件
- "run" 按钮获得 SVM 模型
- "clear" 按钮清空窗口

可以在底部窗口输入选项，选项语法和 `svm-train` 一样。

"load" 和 "save" 在分类和回归中都考虑密集数据格式。
- 对分类，每个数据点有一个标签，值为1, 2 或 3；和两个属性（x-axis 和 y-axis 值），在 [0, 1) 之间；
- 对回归，每个数据点有一个目标值（y-axis）和一个属性（x-axis），范围 [0,1)


## 使用 svm-train
```sh
svm-train [options] training_set_file [model_file]
```
选项：

- `-s svm_type`

设置 SVM 类型，默认 0.
|值|类型|
|---|---|
|0|C-SVC, multi-class classification|
|1|nu-SVC, multi-class classification|
|2|one-class SVM|
|3|epsilon-SVR (regression)|
|4|nu-SVM (regression)|

- `-k kernel_type`  

设置核函数类型，默认2.

|值|类型|
|---|---|
|0|linear, u'*v|
|1|多项式|
|2|RBF|
|3|sigmoid|
|4|precomputed kernel，training_set_file 中的内核值|

其他参数：
|参数|说明|
|---|---|
|`-d degree`|设置核函数的自由度，默认3|
|`-g gamma`|设置核函数的 gamma 值，默认 1/num_features|
|`-r coef0`|设置核函数的 coef0 值，默认0|
|`-c cost`|设置 C-SVC, epsilon-SVR 和 nu-SVR 的参数 C值，默认1|
|`-n nu`|设置 nv-SVC, one-class SVM 和 nu-SVR 的 nu 参数，默认0.5|
|`-p epsilon`|设置 epsilon-SVR loss function 的 epsilon 值，默认0.1|
|`-m cachesize`|设置缓存大小 MB，默认100|
|`-e epsilon`|set tolerance of termination criterion (default 0.001)|
|`-h shrinking`|whether to use the shrinking heuristic, 0 or 1 (default 1)|
|`-b probability_estimates`|whether to train a SVC or SVR for probability estimates, 0 or 1 (default 0)|
|`-wi weight`|set the parameter C of class i to weight*C, for C-SVC (default 1)|
|`-v n`|n-fold cross validation mode|
|`-q`|quiet mode (no outputs)|
|`-v`|随机将数据分成 n 份，计算交叉验证准确度和均方差|

### 输出含义

train 输出内容如下：
```
optimization finished, #iter = 219
nu = 0.431030
obj = -100.877286, rho = 0.424632
nSV = 132, nBSV = 107
Total nSV = 132
```

解释：
|输出|说明|
|---|---|
|obj|dual SVM probelm 的最优目标值|
|rho|the bias term in the decision function sgn(w^Tx - rho)|
|nSV and nBSV|number of support vectors and bounded support vectors (i.e., alpha_i = C)|
|nu-svm|is a somewhat equivalent form of C-SVM where C is replaced by nu|
|nu|simply shows the corresponding parameter|

## svm-predict
```sh
svm-predict [options] test_file model_file output_file
```

选项：

- `-b probability_estimates`  

是否预测概率估计，0 或 1，对 one-class SVM 只能为0。

参数
- model_file 是由 `svm-train` 输出的模型文件
- test_file 是你希望预测的数据
- svm-predict 将结果输出到 output_file 中

## svm-scale
```sh
svm-scale [options] data_filename
```
|选项|说明|
|---|---|
|`-l lower`|x 缩放下限（默认 -1）|
|`-u upper`|x 缩放上限（默认 +1）|
|`-y y_lower y_upper`|设置y缩放范围（默认无缩放）|
|`-s save_filename`|将缩放参数保存在 save_filename|
|`-r restore_filename`|从 restore_filename 恢复缩放参数|

## `grid` 使用
grid.py 是用于RBF核函数 C-SVM 分类参数选择工具。使用交叉验证（CV）技术评价每个参数组合的准确度，找到适合问题的最优参数。

使用：
```
grid.py [grid_options] [svm_options] dataset
```

grid_options :
-log2c {begin,end,step | "null"} : set the range of c (default -5,15,2)
    begin,end,step -- c_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with c
-log2g {begin,end,step | "null"} : set the range of g (default 3,-15,-2)
    begin,end,step -- g_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with g
-v n : n-fold cross validation (default 5)
-svmtrain pathname : set svm executable path and name
-gnuplot {pathname | "null"} :
    pathname -- set gnuplot executable path and name
    "null"   -- do not plot
-out {pathname | "null"} : (default dataset.out)
    pathname -- set output file path and name
    "null"   -- do not output file
-png pathname : set graphic output file path and name (default dataset.png)
-resume [pathname] : resume the grid task using an existing output file (default pathname is dataset.out)
    Use this option only if some parameters have been checked for the SAME data.

svm_options : additional options for svm-train

The program conducts v-fold cross validation using parameter C (and gamma)
= 2^begin, 2^(begin+step), ..., 2^end.

## 使用技巧
具体内容：
- 缩放数据，例如将所有属性缩放至 [0,1] 或 [-1,+1]
- 对 C-SVC，可以使用 tools 目录中的模型选择工具
- nu-SVC/one-class-SVM/nu-SVR 中的 nu 值近似于训练误差和支持向量的比例
- 如果用于分类的训练数据不平衡（如许多正例，负例很少），可以通过 `-wi` 选项尝试使用不同的乘法参数 C 值
- 如果数据量很大，通过 `-m` 设置更大的缓存值。

## 示例
```sh
svm-scale -l -1 -u 1 -s range train > train.scale
svm-scale -r range test > test.scale
```
将训练数据的属性缩放到 [-1,1] 之间。缩放参数保存在 range 文件中，并用来缩放测试数据。

```
svm-train -s 0 -c 5 -t 2 -g 0.5 -e 0.1 data_file
```
使用 RBF 内核 $\exp(-0.5|u-v|^2)$, C=2, stopping tolerance=0.1。

```
svm-train -s 3 -p 0.1 -t 0 data_file
```
使用线性核 u'v，损失函数 epsilon=0.1，用于SVM 回归。

```
svm-train -c 10 -w1 1 -w-2 5 -w4 2 data_file
```
训练分类器，对 class 1 惩罚值 $10=1*10$，对 class -2 惩罚值 $50=5*10$，对分类 4 惩罚值 $20=2*10$

```
svm-train -s 0 -c 100 -g 0.1 -v 5 data_file
```
使用参数 $C=100$ 和 $gamma=0.1$ 对分类任务执行五重交叉验证。

```sh
svm-train -s 0 -b 1 data_file
svm-predict -b 1 test_file data_file.model output_file
```

获得具有概率信息的模型，并使用概率估计测试数据。

## Precomputed Kernels
用户可以先计算内核值，将它们输入到训练和测试文件。此时 libsvm 不需要原来的训练和测试数据集。

假设有 L 个训练样本 $x_1,...,x_L$，令 $K(x,y)$ 为样本 $x$ 和 $y$ 的 kernel 值。

此时，训练集的输入格式为：

```
<label> 0:i 1:K(xi,x1) ... L:K(xi,xL)
```

测试集的输入格式为：

```
<label> 0:? 1:K(x,x1) ... L:K(x,xL)
```

在训练集中，第一列必须是 $x_i$ 的 ID。在测试集中，$?$ 可以是任意值。

必须显式提供所有 kernel 值，包括 0。训练集和测试集的任意重排或随机子集也必须有效。

例如：

假设原训练集 3 个样本，每个 样本 4 个特征，测试集有 1 个样本：

```
15  1:1 2:1 3:1 4:1
45      2:3     4:3
25          3:1

15  1:1     3:1
```



# 准备数据
训练一个大的数据集十分耗时，有时候，我们应该先从一个小的子集开始处理。 `subset.py` 脚本随机
说明：
- LIBSVM 忽略值为 0 的数据；
- LIBSVM 不支持非数值数据，所以需要提前将非数值数据转换为数值数据；

# QA

## 结果重现性
由于数据随机划分问题，不同系统不同时间运行，CV准确度可能会有所不同

## 数据
libsvm 会忽略所有的 0 数据。

## model file
model 文件：
- 开头是参数和标签
- 随后每一行是一个支持向量，按照之前的 label 的顺序排列

## C 值
- 只选择小C值，因为当那个C值大于某个阈值后，其结果和小 C 值一样。

关键的是，C值过大模型不够稳健。
## 值范围
对 linear scaling method，如果使用 RBF 核并且优化的参数，则选择 [0,1] 和 [-1,1] 一样。

## 概率
LIBSVM 使用 cross-validation 计算概率，因此比常规的训练更为耗时。

因此推荐的方式是，先使用无概率的方式选择参数，要选择合适的参数后，再选择输出概率。即在选择参数时不要启用 `-b` 选项，在选择好合适的参数后，不再需要交叉验证时，再使用 `-b` 选项。即 `-b` 和 `-v` 不要同时使用。

## 参考
- https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
- https://github.com/cjlin1/libsvm