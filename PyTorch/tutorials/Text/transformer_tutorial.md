# 基于 nn.Transformer 和 TorchText 的语言模型

- [基于 nn.Transformer 和 TorchText 的语言模型](#基于-nntransformer-和-torchtext-的语言模型)
  - [简介](#简介)
  - [定义模型](#定义模型)
  - [参考](#参考)

***

## 简介

下面介绍使用 [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) 训练 seq2seq 模型。

PyTorch 1.2 根据论文 [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 实现了一个标准 Transformer 模块。与 RNN 相比，Transformer 模型已被证明在许多 seq2seq 任务中质量更好，同时具有更强的并行性。`nn.Transformer` 模块完全依赖于注意力机制（实现为 [nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)）来捕获输入到输出的全局依赖关系。`nn.Transformer` 模块高度模块化，其组件（如 [nn.TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)）可以很容易地修改和组合。

![](images/2022-11-08-18-33-57.png)

## 定义模型

下面在一个语言建模任务上训练 `nn.TransformerEncoder` 模型。**语言模型**是预测单词序列的下一个单词为指定单词的概率。首先将 token 序列传入嵌入层，然后添加位置编码。`nn.TransformerEncoder` 包含多层 [nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)。在输入序列的同时，还需要一个 attention mask，因为 `nn.TransformerEncoder` 中的 self-attention 层只允许注意序列中当前时间步前面的序列。对语言建模，当前时间步后的所有 token 都应该屏蔽。为了获得输出单词的概率分布，需将 `nn.TransformerEncoder` 的输出传入 log-softmax 函数。

```python
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
```

`PositionalEncoding` 注入 token 在序列中的相对或绝对位置信息。位置编码与嵌入维度相同，因此可以相加。上面的 `PositionalEncoding` 使用不同频率的 `sine` 和 `cosine` 函数。



## 参考

- https://pytorch.org/tutorials/beginner/transformer_tutorial.html
