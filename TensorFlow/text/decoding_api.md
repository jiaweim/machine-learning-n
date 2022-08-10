# Decoding API

@author Jiawei Mao
***

## 概述

近年来，有大量研究利用自回归模型进行语言生成。在自回归语言生成中，在时间步 `K` token 的概率分布取决于前面 `K-1` 步模型的 token 预测。对这些模型，解码策略（decoding strategy）如束搜索（Beam search）、贪婪（Greedy）、Top-p 和 Top-k 是模型的关键组成部分，并且在很大程度上影响指定时间步 `K` 生成输出 token 的风格或性质。

例如，**束搜索**通过在每个时间步保存最可能的 `num_beams` 假设，来降低丢失可能的高概率 token 的风险，最终选择总体概率最高的假设。[Murray et al. (2018)](https://arxiv.org/abs/1808.10006) 和 [Yang et al. (2018)](https://arxiv.org/abs/1808.09582)证明束搜索在机器翻译任务中效果很好。**束搜索**和 **Greedy** 策略都有生成重复 token 的可能。

[Fan et. al (2018)](https://arxiv.org/pdf/1805.04833.pdf) 引入了 Top-K 抽样，取 K 个最可能的 token，然后在这 K 个 token 之间重新计算概率质量分布。

[Ari Holtzman et. al (2019)](https://arxiv.org/pdf/1904.09751.pdf) 引入了 Top-p 抽样，该抽样从累计概率加起来为 `p` 的最小 token 集合中进行选择。然后将概率质量分布重新在该集合中分布，这样，token 集合的大小可以动态增加或减小。**Top-p, top-k** 通常用于 story-generation 之类的任务。

Decoding API 提供了一个接口，可以在自回归模型上试验不同的 Decoding 策略。

1. `sampling_module.py` 提供了以下采样策略
   - [top_p](https://arxiv.org/abs/1904.09751): [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L65)
   - [top_k](https://arxiv.org/pdf/1805.04833.pdf): [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L48)
   - Greedy: [github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/sampling_module.py#L26)

2. 在 beam_search.py 中提供束搜索，[github](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/beam_search.py)

## 设置

```powershell
pip install -q -U "tensorflow-text==2.8.*"
```

```powershell
pip install -q tf-models-official==2.7.0
```

```python
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from official import nlp
from official.nlp.modeling.ops import sampling_module
from official.nlp.modeling.ops import beam_search
```

### 在 TF-NLP 初始化采样模块

- **symbols_to_logits_fn**：使用该闭包来调用模型，以预测 `index+1` step 的 logits。该闭包的输入和输出如下

```python
Args:
  1] ids : Current decoded sequences. int tensor with shape (batch_size, index + 1 or 1 if padded_decode is True)],
  2] index [scalar] : current decoded step,
  3] cache [nested dictionary of tensors] : Only used for faster decoding to store pre-computed attention hidden states for keys and values. More explanation in the cell below.
Returns:
  1] tensor for next-step logits [batch_size, vocab]
  2] the updated_cache [nested dictionary of tensors].
```

`cache` 用于加速解码。这里是上述闭包的[参考实现](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/ops/beam_search_test.py#L88)。

- **length_normalization_fn**：使用该闭包返回长度归一化参数

```python
Args: 
  1] length : scalar for decoded step index.
  2] dtype : data-type of output tensor
Returns:
  1] value of length normalization factor.
```

- **vocab_size**：输出词汇表大小。
- **max_decode_length**：解码总的步骤数。
- **eos_id**：如果 batch 中输出的所有解码 id 包含 `eos_id`，则停止解码。
- **padded_decode**：如果在 TPU 上运行，将其设置为 True。True 表示将张量填充到长度 `max_decode_length`
- **top_k**：如果该值 >1，启用 top_k
- **top_p**：如果该值 >0 且 <1.0，启用 top_p
- **sampling_temperature**：用来重新估计 softmax 输出。Temperature 使分布向高概率的 token 倾斜，并降低尾部分布的概率质量。值必须为正数。低温相当于 greedy，使分布更尖锐；而高温使分布更平坦。
- **enable_greedy**：默认为 True，启用 greedy 解码。要尝试其它策略，将此设置为 False。

## 初始化模型超参数

```python
params = {}
params["num_heads"] = 2
params["num_layers"] = 2
params["batch_size"] = 2
params["n_dims"] = 256
params["max_decode_length"] = 4
```

在自回归架构中，如基于 Encoder-Decoder 的 Transformer，cache 用于快速顺序解码。cache 是一个嵌套字典，存储每一层预计算的隐藏状态（self-attention 和 cross-attention blocks 中的 key value）。

## 初始化 cache

```python
cache = {
    "layer_%d"
    % layer: {
        "k": tf.zeros(
            [
                params["batch_size"],
                params["max_decode_length"],
                params["num_heads"],
                int(params["n_dims"] / params["num_heads"]),
            ],
            dtype=tf.float32,
        ),
        "v": tf.zeros(
            [
                params["batch_size"],
                params["max_decode_length"],
                params["num_heads"],
                int(params["n_dims"] / params["num_heads"]),
            ],
            dtype=tf.float32,
        ),
    }
    for layer in range(params["num_layers"])
}
print("cache key shape for layer 1 :", cache["layer_1"]["k"].shape)
```

```txt
cache key shape for layer 1 : (2, 4, 2, 128)
````



## 参考

- https://www.tensorflow.org/text/guide/decoding_api
