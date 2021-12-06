"""
keras 的内置函数可以对原始文本数据进行单词级或字符级的 one-hot 编码，这些函数实现了许多重要的特性：

- 从字符串中取出特殊字符
- 只考虑数据集中前 N 个最常见的字符（避免处理非常大的输入向量空间）
"""

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个分词器（tokenizer），设置为只考虑前 1000 个最常见的单词
tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(samples)  # 构建单词索引

# 将字符串转换为正数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)

# 也可以直接得到 one-hot 二进制表示，这个分词器也支持除 one-hot 编码外的其它向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index  # 找回单词索引

print("Found %s unique tokens." % len(word_index))
