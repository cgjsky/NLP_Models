import torch
import torch.nn as nn
from data import zidian_y,loader
from mask import mask_pad, mask_tril
from model import Transformer
epochs=2
max_len=50
n_vocab=39
model=Transformer()
loss_func = nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
#train
for epoch in range(epochs):
    for i,(x,y) in enumerate(loader):
        #x->[b,50] y->[b,51]
        pred=model(x,y[:,:-1])
        #pred->[b,max_len,n_vocab]
        pred=pred.reshape(-1,n_vocab)
        y=y[:,1:].reshape(-1)
        select=y!=zidian_y['<PAD>']
        pred=pred[select]
        y=y[select]
        
        loss=loss_func(pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i%200==0:
            pred=pred.argmax(1)
            correct=(pred==y).sum().item()
            accuracy=correct/len(pred)
            lr = optim.param_groups[0]['lr']
            print(epoch, i, lr, loss.item(), accuracy)
    sched.step()
