![image-20220414172654984](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220414172654984.png)

# 编码层：

Mulithead attention+FullyconnectedLayer

输入：带位置编码的词向量 [batch_size,max_len,Embedding_size]

输出：[batch_size,max_len,Embedding_size]

每层输入前先clone一份，在后面进行每层的残差连接

## Multihead attention

输入X ，先LayerNorm，通过三个全连接层得到Q,K,V三个矩阵，Q*K得到初始score。

由于词语长短不同，会有padding，对于<PAD>,我们把别的词对于它的注意力设置为0，所以需要一个mask矩阵对score进行处理，mask矩阵是<PAD>的地方设置为-inf，这样softmax处理的时候会变成0

之后对mask处理之后的score进行softmax，映射到0-1，然后乘上V矩阵，这样就是注意力得分

多头注意力就是对Embedding进行拆分，例如32维的词向量，拆成4个头，每个头的长度是8，这样可以划分成子空间，注意力不会互相影响，最后进行拼接。

## FullyconnectedLayer

如果只有MultiHead，输出的只会是attention的线性组合，表达能力有限，

加上全连接层可以变换维度，而全连接层可以自己学习复杂的特征表达，并且激活函数能提供非线性。  

## 之后就是Multihead attention与FullyConnectedLayer的堆叠

# 解码层：

跟上面编码层的两层相同，多了一个y对x的注意力层，Q矩阵是y，K，V矩阵是编码层的输出。

# 位置编码：

![image-20220414175745901](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220414175745901.png)

pos 词的位置，range(max_len)

i 词向量里每一个的位置，range(embedding_size)

选择这个函数是因为我们假设它能让模型很容易地学会通过相对位置来关注，因为对于任何固定的偏移量k，PE pos+k 可以被表示为PE pos 的线性函数。