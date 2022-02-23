from src.nlp.rnn.rnnlm import Rnnlm
import numpy as np
from src.nlp.common.functions import softmax


class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        """

        Parameters
        ----------
        start_id 第1个单词 ID
        skip_ids 单词 ID 列表，指定的单词不被采样，用于排除数据集中的 <unk>, N 等被预处理过的单词
        sample_size 要采样的单词数量

        Returns
        -------

        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)  # 输出各个单词的得分
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
