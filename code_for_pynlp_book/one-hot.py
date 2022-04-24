from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

text=['Time flies like an arrow.','Fruit flies like banana.']
cv=CountVectorizer(binary=True)

one_hot_vec=cv.fit_transform(text).toarray()
print(cv.get_feature_names())
print(cv.vocabulary_)
#统计词频
l =one_hot_vec.sum(axis=0)
d =cv.get_feature_names()
dict={}
for i,j in zip(l,d):
    dict[j]=i
print(dict)
sns.heatmap(one_hot_vec, annot=True,
            cbar=False,
            yticklabels=['Sentence 2'])
#plt.show()

tfidf_vec=TfidfVectorizer()
tfidf=tfidf_vec.fit_transform(text).toarray()
print(tfidf)
sns.heatmap(tfidf, annot=True,
            cbar=False,
            yticklabels=['Sentence 2'])
plt.show()