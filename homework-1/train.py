import os
import torch
import numpy as np
from nn import TransformerLM
from cos_schedule import get_lr
from get_batch import get_batch
from checkpoint import save_checkpoint,load_checkpoint
from losses import cross_entropy
from clip_gradient_norm import clip_gradient_norm 
context_length=256
d_model=512
num_layers=4
attention_head=16
d_ff=1344
batch_size=64
train_step=7000
warmup_step=700
max_lr=6e-4
min_lr=6e-5
theta = 10000.0
vocab_size=10000
max_norm=1.0
weight_decay=0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(base_dir, "data", "train.bin")
    valid_path = os.path.join(base_dir, "data", "valid.bin")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")

    train_data=np.memmap(train_path,dtype=np.uint16,mode='r')
    valid_data=np.memmap(valid_path,dtype=np.uint16,mode='r')
    model=TransformerLM(vocab_size,d_model,attention_head,d_ff,context_length,theta,vocab_size,num_layers,device).to(device)
    trainer=torch.optim.AdamW(model.parameters(),lr=max_lr,weight_decay=weight_decay)
    loss=cross_entropy
    start_iter=0
    if os.path.exists(checkpoint_path):
        start_iter=load_checkpoint(checkpoint_path,model,trainer)
        start_iter+=1
        print("back point success")
    for i in range(start_iter,train_step):
        lr=get_lr(i,max_lr,min_lr,warmup_step,train_step)
        for param_group in trainer.param_groups:
            param_group['lr']=lr
        model.train()
        trainer.zero_grad()
        x,y=get_batch(train_data,batch_size,context_length,device)
        logit=model(x)
        l=loss(logit,y)
        l.backward()
        clip_gradient_norm(model.parameters(),max_norm)
        trainer.step()

        if i%100==0 or i==train_step-1:
            model.eval()
            with torch.no_grad():
                x,y=get_batch(valid_data,batch_size,context_length,device)
                logit=model(x)
                ll=loss(logit,y)
            print(f"train_loss:{l.item():.4f},valid_loss:{ll.item():.4f}")

        if (i%1000==0 and i>0) or i==train_step-1:
            save_checkpoint(model,trainer,i,checkpoint_path)

if __name__ == "__main__":
    main()


