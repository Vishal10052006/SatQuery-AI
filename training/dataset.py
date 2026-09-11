"""LEVIR-CD compatible paired dataset loader with deterministic evaluation tiling."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

_EXT={".png",".jpg",".jpeg",".tif",".tiff"}

def _files(p:Path):
    return {x.stem:x for x in p.iterdir() if x.suffix.lower() in _EXT}

class BitemporalChangeDataset(Dataset):
    def __init__(self,root,split="train",patch_size=256,augment=False):
        self.root=Path(root); self.split=split; self.patch_size=patch_size; self.augment=augment
        base=self.root/split if (self.root/split/"A").is_dir() else self.root
        self.a=_files(base/"A"); self.b=_files(base/"B"); self.y=_files(base/"label")
        self.keys=sorted(set(self.a)&set(self.b)&set(self.y))
        if not self.keys: raise ValueError(f"No paired A/B/label images found under {base}")
        self.samples=[]
        if split=="train": self.samples=[(k,None,None) for k in self.keys]
        else:
            for k in self.keys:
                with Image.open(self.y[k]) as im: h,w=im.size[1],im.size[0]
                s=self.patch_size
                if h<=s and w<=s: self.samples.append((k,0,0)); continue
                for top in range(0,max(1,h-s)+1,s):
                    for left in range(0,max(1,w-s)+1,s):
                        self.samples.append((k,min(top,max(0,h-s)),min(left,max(0,w-s))))
    def __len__(self): return len(self.samples)
    @staticmethod
    def _rgb(path): return np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)/255.0
    @staticmethod
    def _mask(path): return (np.asarray(Image.open(path).convert("L"))>127).astype(np.float32)
    def _crop(self,a,b,y,top=None,left=None):
        h,w=y.shape; s=self.patch_size
        if h<s or w<s:
            ph=max(0,s-h); pw=max(0,s-w)
            a=np.pad(a,((0,ph),(0,pw),(0,0)),mode="reflect"); b=np.pad(b,((0,ph),(0,pw),(0,0)),mode="reflect")
            y=np.pad(y,((0,ph),(0,pw)),mode="constant"); h,w=y.shape
        if top is None:
            top=random.randint(0,h-s) if self.augment else (h-s)//2; left=random.randint(0,w-s) if self.augment else (w-s)//2
        return a[top:top+s,left:left+s],b[top:top+s,left:left+s],y[top:top+s,left:left+s]
    def __getitem__(self,i):
        k,top,left=self.samples[i]
        a,b,y=self._rgb(self.a[k]),self._rgb(self.b[k]),self._mask(self.y[k])
        a,b,y=self._crop(a,b,y,None if self.split=="train" else top,None if self.split=="train" else left)
        if self.augment and self.split=="train":
            if random.random()<.5: a=a[:,::-1].copy(); b=b[:,::-1].copy(); y=y[:,::-1].copy()
            if random.random()<.5: a=a[::-1].copy(); b=b[::-1].copy(); y=y[::-1].copy()
            r=random.randrange(4)
            if r: a=np.rot90(a,r).copy(); b=np.rot90(b,r).copy(); y=np.rot90(y,r).copy()
        return {"before":torch.from_numpy(a.transpose(2,0,1)).float(),"after":torch.from_numpy(b.transpose(2,0,1)).float(),"mask":torch.from_numpy(y[None]).float(),"id":f"{k}:{top}:{left}"}
