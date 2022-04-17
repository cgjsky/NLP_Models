# 介绍

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

文本分类问题，抽取文本特征，然后转化成固定维度的特征向量，训练分类器

TextCNN通过一维卷积来获取句子的**n-gram**特征表示

TextCNN对文本浅层特征的抽取能力很强，在短文本领域如搜索、对话领域专注于意图分类时效果很好，应用广泛，且速度快，一般是首选。

对长文本领域，TextCNN主要靠filter窗口抽取特征，在长距离建模方面能力受限，且对语序不敏感



什么是n-gram模型？

**N-Gram是一种基于统计语言模型的算法**。

它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。

每一个字节片段称为gram，对所有gram的出现频度进行统计，并且按照事先设定好的阈值进行过滤，形成关键gram列表，也就是这个文本的向量特征空间，列表中的每一种gram就是一个特征向量维度。

该模型基于这样一种假设，第N个词的出现只与前面N-1个词相关，而与其它任何词都不相关，整句的概率就是各个词出现概率的乘积。这些概率可以通过直接从语料中统计N个词同时出现的次数得到。常用的是二元的Bi-Gram和三元的Tri-Gram。

[n-gram讲解](https://zhuanlan.zhihu.com/p/32829048)

# 预训练的 Word Embeddings

Word embedding 是NLP中一组语言模型和特征学习技术的总称。

一般我们都会选择one-hot编码来表示神经网路的输入

但是对于词语，one-hot编码的维度过大，所以需要embedding降低纬度

Embedding是通过网络学习而来的特征表达。简单说就是通过某种方法对原来的One-Hot单词表示的空间映射到另外一个空间：这个空间的单词向量不在是One-Hot形式即0-1表示某类，而是使用一个浮点型的向量表示；新的空间单词向量的维度一般会更小；语义上相近的单词会更加接近。

相当于模型训练one-hot编码，我们不关心output，只关心模型的权重，权重就是embedding。

[Tensorflow——word embedding](https://www.tensorflow.org/text/guide/word_embeddings)

# 卷积

相比于一般CNN中的卷积核，这里的卷积核的宽度一般需要个词向量的维度一样，卷积核的高度则是一个超参数可以设置，比如设置为2、3等。然后剩下的就是正常的卷积过程。

![image-20220324214711388](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220324214711388.png)

正常情况下，卷积核的size是h*k，h是我们自己设定的参数（1-10），短文本选小的，长文本选大的，另一个维度k是已经被embedding的长度所固定的

相当于二维卷积，向两个方向滑动，而一维卷积则只能向下滑动，每次滑动取得一点特征

![image-20220324214953151](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220324214953151.png)

整个过程从X1开始，从[x1:xh]到[x n-h+1,xn] ,最后建造了一个特征图

![image-20220324215121368](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220324215121368.png)

由于我们选择的h不一致，所以我们产生的特征c也不一致，这个问题，我们可以通过最大池化层来进行解决，对每个特征图，我们提取一个最大特征c max

# 其他

Dropout 0-0.5,特征图多，效果不好的时候，尝试加大dropout

注意dropout对test data的影响



```python
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        #原始维度（词汇表里词的总数）-> 降维后的维度 [6,3,3]->[6,3,4]
        self.W = nn.Embedding(vocab_size, embedding_size)
        #全连接层，总的卷积树映射到标签个数
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        #[2,2,2]的卷积核各三个，batch_size=6,input_channel=1,filters_size=2*4
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X)
        #加入一个通道维度1
        embedded_chars = embedded_chars.unsqueeze(1) 

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            #[6,1,3,4]->[6,3,2,1]
            h = F.relu(conv(embedded_chars))
            #构造一个[2,1]的最大池化层,[6,3,2,1]->[6,3,1,1]
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            #[6,3,1,1]->[6,1,1,3]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        #拼接三个卷积核的结果,[6,1,1,3]->[6,1,1,9]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) 
        #flatten以便全连接[6,9]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) 
        model = self.Weight(h_pool_flat) + self.Bias 
        return model

if __name__ == '__main__':
    #词向量长度
    embedding_size = 2 
    #词的长度
    sequence_length = 3 
    #类别
    num_classes = 2 
    #卷积核的size
    filter_sizes = [2,3,4]
    #每个卷积核的数量
    num_filters = 3 


    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    print(inputs)
    targets = torch.LongTensor([out for out in labels]) # To using Torch Softmax Loss function

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,"is Bad Mean...")
    else:
        print(test_text,"is Good Mean!!")
```

