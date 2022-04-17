import torch.nn as nn
import torch
from data import zidian_x,zidian_y

def mask_pad(data):
    #未embedding的词向量,[b,50]
    mask=data==zidian_x["<PAD>"]
    mask=mask.reshape(-1,1,1,50)
    # 复制n次
    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask=mask.expand(-1,1,50,50)
    return mask

def mask_tril(data):
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
    mask=data==zidian_y["<PAD>"]
    mask=mask.reshape(-1,1,50)
    mask=mask+tril
    mask=mask>0
    mask=(mask==1).unsqueeze(dim=1)
    return mask