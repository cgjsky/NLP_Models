from turtle import forward
from regex import F
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
#get_zidian
zidian={}
with open("zidian.txt") as f:
    for line in f.readlines():
        k,v=line.split(" ")
        zidian[k]=int(v)
#print(zidian["<PAD>"])


class my_dataset(Dataset):
    def __init__(self):
        self.data=pd.read_csv("1.txt",nrows=2000)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data.iloc[i]
#Dataloader get data
data=my_dataset()

def collate_fn(data):
    b=len(data)
    #print(b)
    xs=np.zeros((2*b,30))
    for i in range(b):
        s1=data[i][1]
        s2=data[i][2]

        #word_length=30
        s1=[zidian["<SOS>"]]+s1.split(",")[:28]+[zidian["<EOS>"]]+[zidian["<PAD>"]]*28
        xs[i]=s1[:30]
        
        s2=[zidian["<SOS>"]]+s2.split(",")[:28]+[zidian["<EOS>"]]+[zidian["<PAD>"]]*28
        xs[b+i]=s2[:30]
    return torch.LongTensor(xs)

dataloader=DataLoader(dataset=data,batch_size=8,shuffle=True,collate_fn=collate_fn,drop_last=True)
for i,sample in enumerate(dataloader):
    break
#print(sample.shape) [16,30]

#Model
class ForwardBackward(nn.Module):
    def __init__(self,flip):
        super().__init__()
        #embedding_size=256
        self.rnn1=nn.LSTM(input_size=256,hidden_size=256,batch_first=True)
        self.rnn2=nn.LSTM(input_size=256,hidden_size=256,batch_first=True)
        self.fc=nn.Linear(256,out_features=4300)
        self.flip=flip
    def forward(self,x):
        #[16,29,256]
        b=x.shape[0]
        #初始化记忆
        h=torch.zeros(1,b,256)
        c=torch.zeros(1,b,256)
        #顺序运算,维度不变
        #[16,29,256] -> [16,29,256]
        #如果是反向传播,把x逆序
        if self.flip:
            x = torch.flip(x, dims=(1, ))
        out1,(h,c)=self.rnn1(x,(h,c))
        out2,(h,c)=self.rnn2(x,(h,c))

        if self.flip:
            x = torch.flip(x, dims=(1, ))
            out1 = torch.flip(out1, dims=(1, ))
            out2 = torch.flip(out2, dims=(1, ))
        out3=self.fc(out2)
        return x,out1,out2,out3
#x=torch.FloatTensor(16,29,256)
#out = ForwardBackward(flip=True)(x)
#print(out[3].shape)

class ELMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(embedding_dim=256,padding_idx=0,num_embeddings=4300)
        self.fw=ForwardBackward(True)
        self.bw=ForwardBackward(False)
    def forward(self,x):
        #[16,30]
        x=self.embedding(x)
        out_f=self.fw(x[:,:-1,:])
        out_b=self.bw(x[:,1:,:])
        return out_f,out_b
#out=ELMO()(sample)
#print(len(out))
#print(out[0][-1].shape)
model=ELMO()
criteration=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(1):
    for i,x in enumerate(dataloader):
        optim.zero_grad()
        outs_f, outs_b = model(x)
        #计算loss只需要fc的loss
        #[16,29,4300]
        outs_f = outs_f[-1]
        outs_b = outs_b[-1]
        #[16,30]->[16,29]
        x_f=x[:,1:]
        x_b=x[:,:-1]
        #[16,29,4300]->[16*29,4300]
        outs_f = outs_f.reshape(-1, 4300)
        outs_b = outs_b.reshape(-1, 4300)
        #[16,29]->[16*29]
        x_f = x_f.reshape(-1)
        x_b = x_b.reshape(-1)

        loss_f = criteration(outs_f, x_f)
        loss_b = criteration(outs_b, x_b)
        loss = (loss_f + loss_b) / 2

        loss.backward()
        optim.step()

        if i % 20 == 0:
            #统计正确率
            correct_f = (x_f == outs_f.argmax(axis=1)).sum().item()
            correct_b = (x_b == outs_b.argmax(axis=1)).sum().item()
            total = x.shape[0] * 29
            print(epoch, i, loss.item(), correct_f / total, correct_b / total)
def get_emb(x):
    outs_f, outs_b = model(x)
    
    outs_f = outs_f[1]
    outs_b = outs_b[1]

    #正向和反向的输出不能对齐,把他们重叠的部分截取出来[16,28,256]
    outs_f = outs_f[:, 1:]
    outs_b = outs_b[:, :-1]

    #拼合在一起,就是编码结果了
    #[16,28,256 + 256]
    embed = torch.cat((outs_f, outs_b), dim=2)

    #[16,28,512]
    return embed
get_emb(sample).shape





