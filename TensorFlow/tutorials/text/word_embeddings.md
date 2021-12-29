# 词嵌入

- [词嵌入](#词嵌入)
  - [以数字表示文本](#以数字表示文本)
    - [one-hot encodings](#one-hot-encodings)
    - [每个单词一个唯一编号](#每个单词一个唯一编号)
    - [Word embeddings](#word-embeddings)
  - [参考](#参考)

2021-12-29, 17:35
@author Jiawei Mao
***

## 以数字表示文本

机器学习模型以向量（数组）作为输入，在处理文本时，就需要一种将文本转换为数字的方法。下面我们介绍实现这一目标的三种策略。

### one-hot encodings

首先想到的，可能就是独热编码（one-hot）。考虑如下一句话 "The cat sat on the mat"。这个句子的词汇（vocabulary，即 unique 单词）为 (cat, mat, on, sat, the)。为了表示这些单词，可以创建一个长度等于词汇表的零向量，然后将单词对应的位置设置为 1。如下图所示：

![](images/2021-12-29-17-41-56.png)

要将句子转换为向量，可以将单词的 one-hot 向量串在一起。

> 要点：这个方法效率很低，one-hot 向量是稀疏的（一个 1，其它都是 0）。如果词汇表长度为 10,000，那每个单词的 one-hot 编码 99.99% 的值都是 0.

### 每个单词一个唯一编号

对每个单词进行编号，例如，将 1 分配给 "cat"，2 分配给 "mat" 等等。这样就可以将 "The cat sat on the mat" 编码为一个稠密 vector [5, 1, 4, 3, 5, 2]。这个方法获得的向量不再是稀疏向量，所有元素都包含值。

然而，这个方法有两个缺点：

- 使用整数编码太随意的，不能捕获任何单词之间的关系；
- 对模型来说，整数编码很难获得好模型。例如，一个线性分类器，为每个 feature 学习一个 weight，由于两个单词及它们的编码没有任何相似度可言，这种 feature-weight 的组合没有任何意义。

### Word embeddings

词嵌入（word embeddings）为我们提供您了 一种使用高效、密集表示单词的方法，在这种表示中，相似的单词具有相似的编码。最重要的是，你不需要手动指定这种编码。嵌入（embedding）

## 参考

- https://www.tensorflow.org/text/guide/word_embeddings
