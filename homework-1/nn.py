import torch
import torch.nn as nn
import math
from einops import rearrange


class Linear(nn.Module):
    def __init__(self,input_layer,output_layer,device=None,dtype=None):
        super().__init__()
        factory_keys={'device':device,'dtype':dtype}
        self.w=nn.Parameter(torch.empty(output_layer,input_layer,**factory_keys))
        std=(2.0/(input_layer+output_layer))**0.5
        nn.init.trunc_normal_(self.w,mean=0.0,std=std,a=-3*std,b=3*std)
    def forward(self,x):
        return x@self.w.T

class Embedding(nn.Module):
    def __init__(self,max_len,embedding_size,device=None,dtype=None):
        super().__init__()
        factory_keys={'device':device,'dtype':dtype}
        self.w=nn.Parameter(torch.empty(max_len,embedding_size,**factory_keys))
        #std=(2.0/(input_layer+output_layer))**0.5
        nn.init.trunc_normal_(self.w,mean=0.0,std=1.0,a=-3.0,b=3.0)
    def forward(self,x):
        return self.w[x]

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5,device=None,dtype=None):
        super().__init__()
        factory_keys={'device':device,'dtype':dtype}
        self.t1=nn.Parameter(torch.ones(d_model,**factory_keys))
       # self.t2=nn.Parameter(torch.zeros(d_model,**factory_keys))
        self.eps=eps
    def forward(self,x):
        in_dtype=x.dtype
        x_float=x.to(torch.float32)
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
        result=x_float/rms*self.t1
        return result.to(in_dtype)

def silu_fn(x):
    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,)


