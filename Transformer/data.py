import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
#n_vocab=39
zidian_x='<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
zidian_x={word:i for i,word in enumerate(zidian_x.split(","))}
zidian_y={k.upper():v for k,v in zidian_x.items()}
def get_data():
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r',
        't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',
        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']
    #len(words)=36
    max_len=50
    p=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    p=p/p.sum()
    #随机产生词
    n=random.randint(30,48)
    x=np.random.choice(words,size=n,replace=True,p=p)
    x=x.tolist()
    def get_y(i):
        i=i.upper()
        if not i.isdigit():
            return i
        i=9-int(i)
        return str(i)
    y=[get_y(i) for i in x]
    y=y+[y[-1]]
    y=y[::-1]
    x=["<SOS>"]+x+["<EOS>"]
    y=["<SOS>"]+y+["<EOS>"]
    x=x+["<PAD>"]* 50
    y=y+["<PAD>"]* 51   
    x=x[:50]
    y=y[:51]

    x=[zidian_x[i] for i in x]
    y=[zidian_y[i] for i in y]
    x=torch.LongTensor(x)
    y=torch.LongTensor(y)
    return x,y
#数据集
class x_y_Dataset(Dataset):
    def __init__(self):
        super(Dataset,self).__init__()
    def __len__(self):
        return 10000
    def __getitem__(self, i):
        return get_data()

loader=DataLoader(dataset=x_y_Dataset(),batch_size=8, drop_last=True,shuffle=True,collate_fn=None)


