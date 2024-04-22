from .model import Unet
import torch
from collections import OrderedDict
import sys
def align_checkpoint_dict(pretrain_weight_name):
    checkpoint =torch.load(pretrain_weight_name + '.pth')
    # standardize the checkpoint keys
    cl_head_checkpoint=OrderedDict()      

    # filter out the decoder layers checkpoint
    if 'encoder.backbone.conv1.weight' in checkpoint.keys() or 'encoder.backbone.features.0.weight' in checkpoint.keys():
        print("Loading pretrain weights for UNet from OLD repo unrolled_lrp pretrain")
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: #or 'classifier' in key:
                del checkpoint[key]
            elif 'encoder.backbone.' in key:
                
                checkpoint[key.replace('encoder.backbone.', '')] = checkpoint[key]
                del checkpoint[key]

        # fill in the classification head checkpoint
        for key in list(checkpoint.keys()):
            if 'classifier' in key  or 'fc' in key:
                cl_head_checkpoint[key]=checkpoint[key]
                del checkpoint[key]

    elif 'encoder.conv1.weight' in checkpoint.keys() or 'encoder.features.0.weight' in checkpoint.keys():
        print("Loading pretrain weights for UNet from NEW repo unrolled_lrp pretrain")
        for key in list(checkpoint.keys()):
            if 'tied_weights_decoder' in key: #or 'classifier' in key:
                del checkpoint[key]
            elif 'encoder' in key:
                
                checkpoint[key.replace('encoder.', '')] = checkpoint[key]
                del checkpoint[key]

        # fill in the classification head checkpoint
        for key in list(checkpoint.keys()):
            if 'classifier' in key  or 'fc' in key:
                cl_head_checkpoint[key]=checkpoint[key]
                del checkpoint[key]

    return checkpoint,cl_head_checkpoint
    

def load_pretrain_model(model,pretrain_weight_name):
   
    print(f"Load pretrained weights {pretrain_weight_name + '.pth'}")
    checkpoint,cl_head_checkpoint=align_checkpoint_dict(pretrain_weight_name)
    model.encoder.load_state_dict(checkpoint)
    if model.classification_head is not None:
        model.classification_head.load_state_dict(cl_head_checkpoint)

    return model