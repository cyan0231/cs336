import torch
import numpy as np


def get_batch(dataset,batch_size,context_long,device):
    n=len(dataset)
    max_idx=n-context_long-1
    ix=torch.randint(0,max_idx+1,(batch_size,))
    x=[dataset[i:i+context_long] for i in ix]
    y=[dataset[i+1:i+1+context_long] for i in ix]
    x=torch.from_numpy(np.array(x)).to(device).long()
    y=torch.from_numpy(np.array(y)).to(device).long()
    return x,y