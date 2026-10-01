import torch
import os
import typing

def save_checkpoint(model,optimizer,train_step,out_root):
    checkpoint={
        'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'train_step':train_step

    }
    torch.save(checkpoint,out_root)


def load_checkpoint(root,model,optimizer):
    checkpoint=torch.load(root,map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['train_step']
