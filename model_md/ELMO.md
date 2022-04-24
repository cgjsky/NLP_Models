# ELMO

## Embeddings from Language Models

LSTM版本的BERT，encoder———>LSTM 作为第一个提出来的预训练model

## 与Word2Vec的区别：

Word2Vec通过大规模语料对每个单词训练出固定词向量，但没有办法解决多义词的问题

ELMO的核心是给予了每个token一个Word Embedding，即每个句子中样貌相同的词汇也会有不同的Embedding，结合了文本的上下文信息

这里其实就用到了迁移学习的思想，使用了在大规模语料库上训练好的Word Embedding，输入ELMO模型中进行Fine-Tuning，这里ELMo模型的训练数据是去除标签的，可以根据上下文信息学习到当前语境下的Word Embedding。之后的BERT也是类似

ELMO是基于LM的，（n-gram马尔可夫），BERT则是MLM

ELMO双向，如果输入相同的一个token之后，输出Embedding不一致，说明这两个token所处的文本信息不一致，考虑了文本信息

## MODEL

主要model的层就是正反向的LSTM堆叠，且需要同时获取两个方向的LSTM的output

由于正向的由前(0,k-1)预测k，反向是由后(k+1，n)预测k，所需在最后的embedding拼接的时候，只拼接交叉的部分，也就是word_length-2。

假如我们的下游任务是QA，我们将下游任务的输入句子X输入到预训练好的ELMO中，于是X中的每个单词可以得到3个embedding：nn.Embedding,  output_f,  output_b

三个word embedding通过三个不同的权重组合成每个token的新的embedding，权重可以通过学习得来。然后将整合后的这个Embedding作为X句在下游任务中对应单词的输入，该组合后新的embedding作为补充特征给下游任务使用。
