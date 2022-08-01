# seq2seq 学习加法

## 简介

在本例中，训练模型学习两个数字字符串的加法。例如：

- 输入："535+61"
- 输出："596"

可以选择将输入调换（reversed），这样可以提高性能。

理论上，调换输入序列的顺序引入了 source 和 target 的短期依赖关系。

模型：

- 两个数字（reversed）：一个 LSTM（128 HN），5k 训练样本 = 99% train/test 精度 in 55 epochs
- 三个数字（reversed）：一个 LSTM（128 HN），50k 训练样本 = 99% train/test 精度 in 100 epochs
- 四个数字（reversed）：一个 LSTM（128 HN），400k 训练样本 = 99% train/test 精度 in 20 epochs
- 五个数字（reversed）：一个 LSTM（125 HN），550k 训练样本 = 99% train/test 精度 in 30 epochs

## 配置



## 参考

- https://keras.io/examples/nlp/addition_rnn/
