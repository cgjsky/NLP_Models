# Bert

多层双向transformer编码器➕MLM（masked language model）和NSP（next sequen prediction）两个预训练任务

## 输入

3个Embedding层的加和

![image-20220414221756368](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220414221756368.png)

- Token Embeddings： 采用 **WordPiece** embeddings (Wu et al.,2016) with a 30,000 token vocabulary. 并且使用#划分词改善稀有词的表示，详见Google's Neural Machine Translation System
- Segment Embeddings：用于区分两个句子
- Position Embeddings：位置编码，老生常谈transformer没有捕捉位置信息的能力，所以需要额外的位置编码，这里没有使用transformer论文中的正弦位置编码， 而是采用了learned positional embeddings。

句首第一个token [CLS]：分类用的

句尾的分隔符[SEP]

## 预训练任务

### 1.MLM掩码语言模型

随机mask掉一定比例的input tokens，然后只预测这些被mask掉的tokens。这就是所谓的 “masked LM” (MLM)，相当于完形填空！

存在问题：

- 缺陷1：MASK造成了预训练与微调过程的失配，为了缓解这个问题：并不总是将“masked” words 替换为真正的 [MASK]标记，而是在训练时随机选择15%的tokens，然后按照8:1:1的比例做以下三种操作：（1）替换为[MASK] token；（2）替换为随机词；（3）不变。
- 缺陷2：每个batch只预测15%的tokens，需要更多的轮次才能收敛。这个问题不大，性能的提升胜过了这点儿训练成本。

### 2. NSP下句预测

针对许多重要的下游任务，例如问答（QA）和自然语言推断（NLI）

对于训练语料中一对句子A和B，B有一半的概率是A的下一句，一半的概率是随机的句子。



## fine-tuning

## CLS

Bert在第一句前加一个[CLS]用于分类任务，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。

### 为什么选它表示整句话语义？

因为与文本的其他词相比，这个无明显语义信息的符号会更加公平的融合文本中各个词语的语义信息，从而更好的表达整句话的含义

具体来说，self-attention是用文本中的其它词来增强目标词的语义表示，但是目标词本身的语义还是会占主要部分的（残差连接）

因此，经过BERT的12层，每次词的embedding融合了所有词的信息，可以去更好的表示自己的语义。

而[CLS]位本身没有语义，经过12层，得到的是attention后所有词的加权平均，相比其他正常词，可以更好的表征句子语义。

当然，也可以通过对最后一层所有词的embedding做pooling去表征句子语义。

