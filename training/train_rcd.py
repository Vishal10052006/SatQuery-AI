"""Train/evaluate the SatQuery Siamese U-Net change detector."""
from __future__ import annotations
import argparse,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.change.siamese_unet import SiameseUNet
from training.dataset import BitemporalChangeDataset

def dice_loss(logits,target):
    p=torch.sigmoid(logits); inter=(p*target).sum((1,2,3)); den=p.sum((1,2,3))+target.sum((1,2,3)); return (1-(2*inter+1)/(den+1)).mean()
def metrics(logits,target):
    p=(torch.sigmoid(logits)>=.5); y=target>=.5; tp=(p&y).sum().item(); fp=(p&~y).sum().item(); fn=(~p&y).sum().item(); i=tp/(tp+fp+fn+1e-9); f=2*tp/(2*tp+fp+fn+1e-9); return f,i

def run_epoch(model,loader,opt,device,train):
    model.train(train); total=f1=iou=0; n=0
    for batch in loader:
        a=batch["before"].to(device); b=batch["after"].to(device); y=batch["mask"].to(device); logits=model(a,b); loss=nn.functional.binary_cross_entropy_with_logits(logits,y)+dice_loss(logits,y)
        if train: opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        f,i=metrics(logits.detach(),y); bs=a.size(0); total+=loss.item()*bs; f1+=f*bs; iou+=i*bs; n+=bs
    return total/max(1,n),f1/max(1,n),iou/max(1,n)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--patch-size",type=int,default=256); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--checkpoint",default="checkpoints/rcd/building_change_siamese_unet.pth"); ap.add_argument("--workers",type=int,default=2); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}")
    tr=BitemporalChangeDataset(args.data_root,"train",args.patch_size,True); va=BitemporalChangeDataset(args.data_root,"val",args.patch_size,False)
    tl=DataLoader(tr,args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=="cuda"); vl=DataLoader(va,args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda")
    model=SiameseUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); best=-1; Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        loss,_,_=run_epoch(model,tl,opt,device,True); vloss,vf1,viou=run_epoch(model,vl,opt,device,False); print(f"epoch={epoch:03d} train_loss={loss:.4f} val_loss={vloss:.4f} val_f1={vf1:.4f} val_iou={viou:.4f}")
        if vf1>best:
            best=vf1; torch.save({"model_state_dict":model.state_dict(),"model_config":{"in_channels":3,"base":32},"epoch":epoch,"val_f1":vf1,"val_iou":viou},args.checkpoint)
    print(f"best_checkpoint={args.checkpoint} val_f1={best:.4f}")

if __name__=="__main__": main()
