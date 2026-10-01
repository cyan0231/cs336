import torch


def clip_gradient_norm(parameters,max_norm):
    grad_ok=[p for p in parameters if p.grad is not None]
    if not grad_ok:
        return
    eps=1e-6
    total_grad=0.0
    for p in grad_ok:
        param_norm=torch.norm(p.grad.detach(),p=2)
        total_grad+=param_norm.item()**2
    total_grad=total_grad**0.5
    if total_grad>max_norm:
        rate=max_norm/(total_grad+eps)
        for p in grad_ok:
            p.grad.detach().mul_(rate)
        
