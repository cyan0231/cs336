import torch

def cross_entropy(logit,target):
    target_logit=torch.gather(logit,dim=-1,index=target.unsqueeze(-1)).squeeze(-1) #batch step
    m=torch.max(logit,dim=-1,keepdim=True).values #batch step 1
    t=logit-m #batch step vocab_size
    t=m+torch.log(torch.sum(torch.exp(t),dim=-1,keepdim=True))
    loss=t.squeeze(-1)-target_logit
    return torch.mean(loss)

