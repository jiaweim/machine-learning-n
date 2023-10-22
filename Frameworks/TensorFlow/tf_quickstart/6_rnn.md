# 循环神经网络实现连续数据的预测

- [循环神经网络实现连续数据的预测](#循环神经网络实现连续数据的预测)
  - [循环神经网络](#循环神经网络)
    - [循环核](#循环核)
    - [循环核按时间步展开](#循环核按时间步展开)
    - [循环计算层](#循环计算层)
    - [TF 描述循环计算层](#tf-描述循环计算层)
    - [循环计算过程](#循环计算过程)
  - [实践：ABCDE 字母预测](#实践abcde-字母预测)
    - [one-hot](#one-hot)
    - [Embedding](#embedding)
  - [实践：股票预测](#实践股票预测)
    - [RNN](#rnn)
    - [LSTM](#lstm)
    - [GRU](#gru)

***

## 循环神经网络

### 循环核

循环核：参数时间共享，循环层提取时间信息。

![](2022-10-27-13-53-19.png)

### 循环核按时间步展开

![](2022-10-27-14-09-48.png)

### 循环计算层

![](images/2022-10-27-14-09-22.png)

### TF 描述循环计算层

```python
tf.keras.layers.SimpleRNN(
    记忆体个数，
    activation=‘激活函数’ ，
    return_sequences=是否每个时刻输出ht到下一层)
    activation=‘激活函数’ （不写，默认使用tanh）
    return_sequences=True 各时间步输出ht
    return_sequences=False 仅最后时间步输出ht（默认）
```

TF API 对送入循环层的数据维度有要求：

![](2022-10-27-14-22-25.png)

### 循环计算过程

## 实践：ABCDE 字母预测

### one-hot

### Embedding

## 实践：股票预测

### RNN

### LSTM

### GRU
