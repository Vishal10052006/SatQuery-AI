"""Target-guided bi-temporal change detection with semantic neural inference."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import numpy as np
from .mask_processing import ChangeRegion, extract_change_regions
from .preprocess import PreprocessedImage

@dataclass
class RCDResult:
    status:str; model_name:str; detector_type:str; target:str|None; change_mask:np.ndarray|None; difference_map:np.ndarray|None
    regions:list[ChangeRegion]=field(default_factory=list); confidence:float=0.0; message:str=""; error:str|None=None
    def to_dict(self)->dict[str,Any]:
        return {"status":self.status,"model_name":self.model_name,"detector_type":self.detector_type,"target":self.target,"confidence":round(self.confidence,4),"message":self.message,"error":self.error,"num_regions":len(self.regions),"regions":[r.to_dict() for r in self.regions]}

def _land_change_query(target):
    if not target:return False
    return any(x in target.lower() for x in ("land","construction","constructed","building","built-up","built up","urban","road","infrastructure","development","vegetation","agriculture","crop","deforestation","forest"))

def _likely_water_mask(image):
    if image.data.ndim!=3 or image.data.shape[-1]<3:return np.zeros(image.gray.shape,bool)
    r,g,b=image.data[:,:,0],image.data[:,:,1],image.data[:,:,2]; bright=(r+g+b)/3; ratio=b/(r+g+b+1e-6)
    return ((bright<.34)&(b>=r*1.02)&(g>=r*.98))|((ratio>.36)&(b>=r*1.08)&(g>=r*.98))

def _suppress_water_change(diff,before,after,target):
    if not _land_change_query(target):return diff,np.zeros(diff.shape,bool)
    water=_likely_water_mask(before)|_likely_water_mask(after); out=diff.copy(); out[water]=0; return out,water

def _clean_temporal_mask(raw,diff,min_pixels):
    h,w=raw.shape
    if h<256 or w<256:return raw,max(1,int(min_pixels))
    try:
        from scipy.ndimage import binary_closing,binary_opening
        s=np.ones((3,3),bool); clean=binary_closing(binary_opening(raw,s),s)
    except ImportError: clean=raw
    return clean,max(int(min_pixels),int(round(raw.size*.00015)))

class RCDAdapter:
    def __init__(self,checkpoint_path:str|Path|None=None):
        env=os.getenv("SATQUERY_RCD_CHECKPOINT")
        self.checkpoint_path=Path(checkpoint_path or env) if (checkpoint_path or env) else None
        self._model=None; self._load_attempted=False
    def _candidate_paths(self):
        return ([self.checkpoint_path] if self.checkpoint_path else [])+[Path("checkpoints/rcd/building_change_siamese_unet.pth"),Path("checkpoints/rcd_model.pth"),Path("checkpoints/rcd/model.pt"),Path("models/change/weights/rcd.pth")]
    def _existing_checkpoint(self):
        for p in self._candidate_paths():
            if p and p.is_file():return p
        return None
    def is_model_available(self):
        return self._existing_checkpoint() is not None
    def _load_model(self):
        if self._load_attempted:return self._model
        self._load_attempted=True; path=self._existing_checkpoint()
        if not path:return None
        try:
            import torch
            from .siamese_unet import SiameseUNet
            ckpt=torch.load(path,map_location="cpu",weights_only=False)
            state=ckpt.get("model_state_dict",ckpt.get("state_dict",ckpt)) if isinstance(ckpt,dict) else ckpt
            cfg=ckpt.get("model_config",{}) if isinstance(ckpt,dict) else {}
            model=SiameseUNet(in_channels=int(cfg.get("in_channels",3)),base=int(cfg.get("base",32)))
            if any(k.startswith("module.") for k in state): state={k.removeprefix("module."):v for k,v in state.items()}
            model.load_state_dict(state,strict=True); model.eval(); self._model=model
        except Exception:
            self._model=None
        return self._model
    def detect_target_change(self,before,after,target,threshold=.15,min_pixels=8,allow_fallback=True):
        if self.is_model_available():
            result=self._run_neural_inference(before,after,target,min_pixels)
            if result.status=="success":return result
            if not allow_fallback:return result
        if not allow_fallback:
            return RCDResult("awaiting_model","rcd-neural-adapter","adapter",target,None,None,message=f"Referring Change Detection for '{target}' requires a trained neural checkpoint.")
        diff=np.abs(after.gray-before.gray).astype(np.float32); valid=before.valid_mask&after.valid_mask; diff[~valid]=0
        diff,water=_suppress_water_change(diff,before,after,target); raw=diff>=float(threshold); raw[water]=False
        clean,effective=_clean_temporal_mask(raw,diff,min_pixels); mask,regions=extract_change_regions(clean,diff,min_pixels=effective)
        for r in regions:r.target=target
        msg=f"Land-surface temporal change regions surfaced for '{target}' using the deterministic fallback; semantic building/road classification is unavailable." if _land_change_query(target) else f"Temporal change regions surfaced for '{target}' using the deterministic fallback; target semantics were not modeled."
        if np.any(water):msg+=" Persistent water signatures were excluded from the land-change mask."
        return RCDResult("fallback_baseline","rcd-fallback-baseline","fallback_baseline",target,mask,diff,regions,.35,msg)
    def _run_neural_inference(self,before,after,target,min_pixels):
        model=self._load_model()
        if model is None:return RCDResult("awaiting_model","rcd-neural-checkpoint","neural",target,None,None,message="Semantic RCD checkpoint could not be loaded.")
        try:
            import torch
            a=before.data[:,:,:3] if before.data.shape[-1]>=3 else np.repeat(before.data,3,axis=2)
            b=after.data[:,:,:3] if after.data.shape[-1]>=3 else np.repeat(after.data,3,axis=2)
            h,w=a.shape[:2]; ph=(16-h%16)%16; pw=(16-w%16)%16
            if ph or pw:
                a=np.pad(a,((0,ph),(0,pw),(0,0)),mode="edge"); b=np.pad(b,((0,ph),(0,pw),(0,0)),mode="edge")
            with torch.inference_mode():
                logits=model(torch.from_numpy(a.transpose(2,0,1)).float().unsqueeze(0),torch.from_numpy(b.transpose(2,0,1)).float().unsqueeze(0)); prob=torch.sigmoid(logits)[0,0].cpu().numpy()
            prob=prob[:h,:w].astype(np.float32); mask=prob>=.5; mask &= before.valid_mask & after.valid_mask
            clean,regions=extract_change_regions(mask,prob,min_pixels=max(1,min_pixels))
            for r in regions:r.target=target
            detected=float(prob.max())>.5 and bool(clean.any())
            return RCDResult("success","rcd-siamese-unet","neural",target,clean,prob,regions,.0 if not detected else float(prob[clean].mean()),"Semantic bi-temporal building-change segmentation completed.")
        except Exception as exc:
            return RCDResult("awaiting_model","rcd-neural-checkpoint","neural",target,None,None,message="Semantic RCD inference failed; no model-derived mask returned.",error=str(exc))
