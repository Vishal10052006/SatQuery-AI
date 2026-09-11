import numpy as np
import torch
from models.change.siamese_unet import SiameseUNet
from models.change.rcd import RCDAdapter

def test_siamese_unet_forward_shape():
    model=SiameseUNet().eval()
    with torch.no_grad():
        out=model(torch.rand(1,3,64,64),torch.rand(1,3,64,64))
    assert out.shape==(1,1,64,64)

def test_missing_checkpoint_uses_explicit_fallback():
    rcd=RCDAdapter('/tmp/satquery-no-such-checkpoint.pth')
    P=type('P',(),{})
    before=P(); after=P()
    before.gray=np.zeros((32,32),np.float32); after.gray=np.ones((32,32),np.float32)*.5
    before.data=np.zeros((32,32,3),np.float32); after.data=np.ones((32,32,3),np.float32)
    before.valid_mask=after.valid_mask=np.ones((32,32),bool); before.original_shape=after.original_shape=(32,32)
    result=rcd.detect_target_change(before,after,'newly constructed buildings')
    assert result.status=='fallback_baseline'
    assert result.model_name=='rcd-fallback-baseline'
