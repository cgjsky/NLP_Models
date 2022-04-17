from re import X
from turtle import forward
from mask import mask_pad,mask_tril
from util import MultiHead,PositionEmbedding,FullyConnectedOutput
import torch 
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh=MultiHead()
        self.fc=FullyConnectedOutput()
    def forward(self,x,mask):
        #x词向量->[b,50,32]
        score=self.mh(x,x,x,mask)
        out=self.fc(score)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=EncoderLayer()
        self.layer_2=EncoderLayer()
        self.layer_3=EncoderLayer()
    def forward(self,x,mask):
        x=self.layer_1(x,mask)
        x=self.layer_2(x,mask)
        x=self.layer_3(x,mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh_1=MultiHead()
        self.mh_2=MultiHead()
        self.fc=FullyConnectedOutput()
    def forward(self,x,y,mask_pad_x,mask_tril_y):
        y=self.mh_1(y,y,y,mask_tril_y)
        y=self.mh_2(y,x,x,mask_pad_x)
        y=self.fc(y)
        return y
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=DecoderLayer()
        self.layer_2=DecoderLayer()
        self.layer_3=DecoderLayer()
    def forward(self,x,y,mask_pad_x,mask_tril_y):
        y=self.layer_1(x,y,mask_pad_x,mask_tril_y)
        y=self.layer_1(x,y,mask_pad_x,mask_tril_y)
        y=self.layer_1(x,y,mask_pad_x,mask_tril_y)
        return y

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x=PositionEmbedding()
        self.embed_y=PositionEmbedding()
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.fc_out=nn.Linear(32,39)
    def forward(self,x,y):
        #padding->[b,1,50,50] 
        mask_pad_x=mask_pad(x)
        mask_tril_y=mask_tril(y)

        x,y=self.embed_x(x),self.embed_y(y)

        x=self.encoder(x,mask_pad_x)
        y=self.decoder(x,y,mask_pad_x,mask_tril_y)
        y=self.fc_out(y)
        return y



