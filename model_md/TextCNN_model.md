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

