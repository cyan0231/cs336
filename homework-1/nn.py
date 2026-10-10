import torch
import torch.nn as nn
import math



class Linear(nn.Module):
    def __init__(self,input_layer,output_layer,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.w=nn.Parameter(torch.empty(output_layer,input_layer,**factory_kwargs))
        std=(2.0/(input_layer+output_layer))**0.5
        nn.init.trunc_normal_(self.w,mean=0.0,std=std,a=-3*std,b=3*std)
    def forward(self,x):
        return x@self.w.T

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_size,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.w=nn.Parameter(torch.empty(vocab_size,embedding_size,**factory_kwargs))
        #std=(2.0/(input_layer+output_layer))**0.5
        nn.init.trunc_normal_(self.w,mean=0.0,std=1.0,a=-3.0,b=3.0)
    def forward(self,x):
        return self.w[x]

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.t1=nn.Parameter(torch.ones(d_model,**factory_kwargs))
       # self.t2=nn.Parameter(torch.zeros(d_model,**factory_kwargs))
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
    def __init__(self,d_model,dff,device=None,dtype=None):
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.d_model=d_model
        self.dff=dff
        self.w1=Linear(d_model,dff,**factory_kwargs)
        self.w2=Linear(d_model,dff,**factory_kwargs)
        self.w3=Linear(dff,d_model,**factory_kwargs)
    def forward(self,x):
        return self.w3(silu_fn(self.w1(x))*self.w2(x))



class Rope(nn.Module):
    def __init__(self,d_model,theta,context_length,device=None):
        super().__init__()
        self.d_model=d_model
        power=torch.arange(0,d_model,2,device=device).float()/d_model
        freq=1.0/(theta**power)
        t=torch.arange(context_length,device=device).float()
        result=torch.outer(t,freq)
        self.register_buffer("cos_cached", result.cos(), persistent=False)
        self.register_buffer("sin_cached", result.sin(), persistent=False)
    def forward(self,x,token_positions):
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)
        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        output=torch.empty_like(x)
        output[...,0::2]=x_even*cos-x_odd*sin
        output[...,1::2]=x_even*sin+x_odd*cos
        return output

def softmax(x,dim=-1):
    x_max=x.max(dim=dim,keepdim=True).values
    x_stable=x-x_max
    exp_x=torch.exp(x_stable)
    sum_exp=exp_x.sum(dim=dim,keepdim=True)
    return exp_x/sum_exp

def scaled_dot_product_attention(q,k,v,mask=None):
    d_k=q.shape[-1]
    score=q@k.transpose(-2,-1)/math.sqrt(d_k)
    if mask is not None:
        score=score.masked_fill(mask==False,float('-inf'))
    score=softmax(score)
    return score@v

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,head_num,context_length=None,theta=None,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.head_num=head_num
        self.head_size=d_model//head_num
        self.context_length=context_length
        self.theta=theta
        self.w_k=Linear(d_model,d_model,device=device,dtype=dtype)
        self.w_v=Linear(d_model,d_model,device=device,dtype=dtype)
        self.w_q=Linear(d_model,d_model,device=device,dtype=dtype)
        self.w_o=Linear(d_model,d_model,device=device,dtype=dtype)
        if theta is not None and context_length is not None:
            self.rope=Rope(self.head_size,theta,context_length,device=device)
        else:
            self.rope=None
    def split(self,x):
        #x batch step d_model
        batch,step=x.shape[0],x.shape[1]
        x=x.reshape(batch,step,self.head_num,self.head_size)
        x=x.permute(0,2,1,3)
        return x
    def merge(self,x):
        batch,step=x.shape[0],x.shape[2]
        x=x.permute(0,2,1,3)
        x=x.reshape(batch,step,-1)
        return x
    def forward(self,x,token_position):
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        q=self.split(q)
        k=self.split(k)
        v=self.split(v)
        if self.rope is not None:

            q=self.rope(q,token_position)
            k=self.rope(k,token_position)

        mask=torch.tril(torch.ones(x.shape[-2],x.shape[-2],device=x.device,dtype=torch.bool))

        atten_out=scaled_dot_product_attention(q,k,v,mask)

        atten_out=self.w_o(self.merge(atten_out))
        return atten_out
        



class TransformerBlock(nn.Module):
    def __init__(self,d_model,head_num,d_ff,context_length,theta,device=None,dtype=None):
        super().__init__()
        self.attention_module=CausalSelfAttention(d_model,head_num,context_length,theta,device=device,dtype=dtype)
        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2=RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)
    def forward(self,x,token_position):
        x=x+self.attention_module(self.ln1(x),token_position)
        x=x+self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,vocab_size,d_model,head_num,d_ff,context_length,theta,output_size,num_layers,device=None,dtype=None):
        super().__init__()
        
        self.embedding=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.context_length=context_length
        self.net=nn.Sequential(*[TransformerBlock(d_model,head_num,d_ff,context_length,theta,device,dtype) for _ in range(num_layers)])
        self.norm=RMSNorm(d_model,device=device,dtype=dtype)
        self.linear=Linear(d_model,output_size,device=device,dtype=dtype)
    def forward(self,x):
        b,s=x.shape
        x=self.embedding(x)
        token_position=torch.arange(s,device=x.device)
        for layer in self.net:
            x=layer(x,token_position)
        x=self.norm(x)
        return self.linear(x)

    @torch.no_grad()
    def generate(self,prompt,max_answer,eos_token_id=None,temperature=1.0,top_p=1.0,): #复制，截取，温度，top-p，softmax，概率抽样，拼接，判断是否结束
        self.eval()
        copy_prompt=prompt.clone()
        for _ in range(max_answer):
            id_copy=copy_prompt[:,-self.context_length:]
            logit=self.forward(id_copy)
            logit=logit[:,-1,:] #batch vacab

            if temperature < 0:
                raise ValueError("temperature 不能小于 0")

            if temperature == 0:
                next_token = logit.argmax(dim=-1, keepdim=True)
            else:
                logit = logit / temperature

                if top_p != 1.0:
                    logit = self._top_p_filter(logit, top_p)

                probs = softmax(logit)
                next_token = torch.multinomial(probs, 1)

            copy_prompt=torch.cat((copy_prompt,next_token),dim=-1)

            if eos_token_id is not None and(next_token==eos_token_id).all():
                break
        return copy_prompt
    def _top_p_filter(self,logit,top_p): #降序排序，算累积概率分布，超过阈值标为true，右移，将需要溢出的logit值设为-inf
        sorted_value,sorted_pos=torch.sort(logit,descending=True,dim=-1) #logit batch vocab

        cumu=torch.cumsum(softmax(sorted_value,dim=-1),dim=-1)

        sorted_indices=cumu>top_p

        sorted_indices[:,1:]=sorted_indices[:,:-1].clone()
        sorted_indices[:,0]=False

        mask=sorted_indices.scatter(1,sorted_pos,sorted_indices)
        logit=logit.masked_fill(mask,float('-inf'))
        return logit






