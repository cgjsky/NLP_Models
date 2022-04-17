# 词袋模型

文本特征提取有两个非常重要的模型：

- 词集模型：单词构成的集合，集合自然每个元素都只有一个，也即词集中的每个单词都只有一个。
- 词袋模型：在词集的基础上如果一个单词在文档中出现不止一次，统计其出现的次数（频数）。           

两者本质上的区别，词袋是在词集的基础上增加了频率的维度，词集只关注有和没有，词袋还要关注有几个。

```python
from sklearn.feature_extraction.text import CountVectorizer

CountVectorizer(analyzer='word', binary=False, decode_error=...'strict',
        dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
texts=["dog cat fish","dog cat cat","fish bird", 'bird'] 
cv=CountVectorizer()
cv.fit_transform(texts)

print(cv.get_feature_names())
out:['bird', 'cat', 'dog', 'fish']

print(cv_fit.toarray())
out:
[[0 1 1 1]
 [0 2 1 0]
 [1 0 0 1]
 [1 0 0 0]]
#每个单词在每个文本中的词频矩阵

l=cv_fit.toarray().sum(axis=0)
d=cv.get_feature_names()
dict={}
for i,j in zip(l,d):
    dict[j]=i
print(dict)
out:{'bird': 2, 'cat': 3, 'dog': 2, 'fish': 2} #每个词的词频


```

CountVectorize函数比较重要的几个参数为：

- decode_error，处理解码失败的方式，分为‘strict’、‘ignore’、‘replace’三种方式。
- strip_accents，在预处理步骤中移除重音的方式。
- max_features，词袋特征个数的最大值。
- stop_words，判断word结束的方式。
- max_df，df最大值。
- min_df，df最小值 。
- binary，默认为False，当与TF-IDF结合使用时需要设置为True。 本例中处理的数据集均为英文，所以针对解码失败直接忽略，使用ignore方式，stop_words的方式使用english，strip_accents方式为ascii方式。

# TF-IDF模型

TF-IDF模型（term frequency–inverse document frequency，词频与逆向文件频率）

TF-IDF是一种统计方法，用以评估某一字词对于一个文件集或一个语料库的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

如果某个词或短语在一篇文章中出现的频率TF(Term Frequency，词频)，词频高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

TF表示词条在文档d中出现的频率。IDF（inverse document frequency，逆向文件频率）的主要思想是：如果包含词条t的文档越少，也就是n越小，IDF越大，则说明词条t具有很好的类别区分能力。

 在Scikit-Learn中实现了TF-IDF算法，实例化TfidfTransformer即可：

```python
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
#一般使用transformer对上一步取得的toarray矩阵进行处理
c=cv_fit.toarray()
tfidf=transformer.fit_transform(c)
tfidf.toarray() 
```

# 词汇表模型

词袋模型可以很好的表现文本由哪些单词组成，但是却无法表达出单词之间的前后关系，于是人们借鉴了词袋模型的思想，使用生成的词汇表对原有句子按照单词逐个进行编码。

```python
#文本进行序列化编码
from tensorflow.contrib import learn
 
vocal = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=0,vocabulary=None,tokenizer_fn=None)

documents = [
    'this is the first test',
    'this is the second test',
    'this is not a test'
]
vocab = learn.preprocessing.VocabularyProcessor(10)
x = np.array(list(vocab.fit_transform(documents)))

[[1 2 3 4 5 0 0 0 0 0]
 [1 2 3 6 5 0 0 0 0 0]
 [1 2 7 8 5 0 0 0 0 0]]
```



# Word2Vec模型

Word2Vec是Google在2013年开源的一款将词表征为实数值向量的高效工具，采用的模型有CBOW(Continuous Bag-Of-Words，即连续的词袋模型)和Skip-Gram 两种。Word2Vec通过训练，可以把对文本内容的处理简化为K维向量空间中的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似度。因此，Word2Vec 输出的词向量可以被用来做很多NLP相关的工作，比如聚类、找同义词、词性分析等等。

![image-20220409215404072](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220409215404072.png)

CBOW模型能够根据输入周围n-1个词来预测出这个词本身，而Skip-gram模型能够根据词本身来预测周围有哪些词。也就是说，CBOW模型的输入是某个词A周围的n个单词的词向量之和，输出是词A本身的词向量，而Skip-gram模型的输入是词A本身，输出是词A周围的n个单词的词向量。 

 Word2Vec最常用的开源实现之一就是gensim

```python
http://radimrehurek.com/gensim/
pip install --upgrade gensim
sentences = [['first', 'sentence'], ['second', 'sentence']]
model = gensim.models.Word2Vec(sentences, min_count=1)
print model['first'] 
```

其中Word2Vec有很多可以影响训练速度和质量的参数。第一个参数可以对字典做截断，少于min_count次数的单词会被丢弃掉, 默认值为5：

```python
model = Word2Vec(sentences, min_count=10)
```

另外一个是神经网络的隐藏层的单元数，推荐值为几十到几百。事实上Word2Vec参数的个数也与神经网络的隐藏层的单元数相同，比如size=200，那么训练得到的Word2Vec参数个数也是200

**Note:Word2Vec的神经网络参数个数就是词向量的长度**

```python
model=gensim.models.Word2Vec(size=200, window=8, min_count=10, iter=10, workers=cores)
```

创建字典并开始训练获取Word2Vec。gensim的官方文档中强调增加训练次数可以提高生成的Word2Vec的质量，可以通过设置epochs参数来提高训练次数，默认的训练次数为5：

```
x=x_train+x_test
model.build_vocab(x)
model.train(x, total_examples=model.corpus_count, epochs=model.iter)
```

```python
def getVecsByWord2Vec(model, corpus, size):
    x=[]
    for text in corpus:
        xx = []
        for i, vv in enumerate(text):
            try:
                xx.append(model[vv].reshape((1,size)))
            except KeyError:
                continue
        x = np.concatenate(xx)
    x=np.array(x, dtype='float')
    return x
```

