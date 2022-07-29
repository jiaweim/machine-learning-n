from src.nlp.rnn.rnnlm import Rnnlm
from src.nlp.dataset import ptb
from src.nlp.common.optimizer import SGD
from src.nlp.common.trainer import RnnlmTrainer
from src.nlp.common.util import eval_perplexity

# 设定超参数
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNN的隐藏状态向量的元素个数
time_size = 35  # RNN的展开大小
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 读入训练数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 生成模型
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 应用梯度裁剪进行学习
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
            eval_interval=20)  # 每 20 次迭代计算一次困惑度
trainer.plot(ylim=(0, 500))

# 基于测试数据进行评价
model.reset_state()  # 需要先重置模型的状态（LSTM 的隐藏状态和记忆单元）
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# 保存参数
model.save_params()
