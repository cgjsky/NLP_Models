from turtle import forward
import torch
import torch.nn as nn
import math

def attention(Q,K,V,mask):
    #[batch_size,head_size,max_len,embedding_size]
    #[b,4,50,8]
    #print(Q.shape)
    score=torch.matmul(Q,K.permute(0,1,3,2))
    score=score/math.sqrt(8)
    score=score.masked_fill_(mask,-float('inf'))
    #缩放到0-1
    score=torch.softmax(score,dim=-1)
    score=torch.matmul(score,V)
    #[b,4,50,8]
    score=score.permute(0,2,1,3).reshape(-1,50,32)
    return score

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q=nn.Linear(32,32)
        self.fc_K=nn.Linear(32,32)
        self.fc_V=nn.Linear(32,32)
        self.fc_out=nn.Linear(32,32)
        self.norm=nn.LayerNorm(normalized_shape=32,elementwise_affine=True)
        self.dropout=nn.Dropout(0.1)
    #传入x,通过三个全连接生成Q，K，V三个矩阵
    def forward(self,Q,K,V,mask):
        #传入的是已经embedding后的词向量,[batch_size,50,32]
        Q=self.norm(Q)
        K=self.norm(K)
        V=self.norm(V)

        batch_size=Q.shape[0]
        #保留一个原始Q以便短接
        clone_Q=Q.clone()

        Q=self.fc_Q(Q)
        K=self.fc_K(K)
        V=self.fc_V(V)
        #多头拆分[batch_size,50,32]->[b,4,50,8]
        Q=Q.reshape(batch_size,50,4,8).permute(0,2,1,3)
        K=K.reshape(batch_size,50,4,8).permute(0,2,1,3)
        V=V.reshape(batch_size,50,4,8).permute(0,2,1,3)
        score=attention(Q,K,V,mask)
        #[b,50,32]
        score=self.fc_out(score)
        score=score+clone_Q
        return score

class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        def get_pe(pos,i,model_dim):
            fenmu=1e4**(i/model_dim)
            pe=pos/fenmu
            if i%2==0:
                return math.sin(pe)
            return math.cos(pe)
        pe=torch.empty(50,32)
        for i in range(pe.shape[0]):
            for j in range(pe.shape[1]):
                pe[i,j]=get_pe(i,j,pe.shape[1])
        pe.unsqueeze(0) #[50,32]->[1,50,32]
        self.register_buffer('pe',pe)
        self.embed = nn.Embedding(39,32)
        #初始化参数
        self.embed.weight.data.normal_(0,0.1)
    def forward(self,x):
        #x--[batch_size,max_len]
        embed=self.embed(x)
        embed=embed+self.pe
        return embed
        
#如果只有MultiHead，输出的只会是attention的线性组合，表达能力有限，
#加上全连接层可以变换维度，而全连接层可以自己学习复杂的特征表达，并且激活函数能提供非线性。  
class FullyConnectedOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(in_features=32,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=32),
            nn.Dropout(0.1)
        )
        self.norm=nn.LayerNorm(normalized_shape=32,elementwise_affine=True)
    def forward(self,x):
        clone_x=x.clone()
        x=self.norm(x)
        out=self.fc(x)
        out=out+clone_x
        return out

