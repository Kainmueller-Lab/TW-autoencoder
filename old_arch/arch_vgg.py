import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as  np
import sys
from .architectures_utils import *
# from .LRP_layers import *
from .AGF_layers import minmax_dims
import importlib
from importlib import import_module as implib
# import importlib.import_module as implib

# --------------------------------------
# Encoder and Decoder
# --------------------------------------

class CNN_Encoder(nn.Module):
    def __init__(self, img_channel=3,input_size=(3, 224, 224),num_classes=6):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size

        self.backbone = initialize_pretrainmodel("vgg16", num_classes, img_channel, feature_extract=False, use_pretrained=True) 
        self.middle_size = self.get_flat_fts([self.backbone.features, self.backbone.avgpool])
        self.flat_fts=int(np.prod(self.middle_size[1:]))


    def get_flat_fts(self, fts):
        if isinstance(fts, (tuple, list)):
            f=Variable(torch.ones(1, *self.input_size))
            for ft in fts:               
                f = ft(f)
        else:
            f = fts(Variable(torch.ones(1, *self.input_size)))
        return f.size()
    
    def forward(self, x): 
        return self.backbone(x) 

    
class CNN_Decoder(nn.Module):
    def __init__(self, encoder, xai, epsilon, alpha,**kwargs):
        super(CNN_Decoder, self).__init__()
        '''
        encoder: nn. Mondule,
        xai: str,
        epsilon: float
        alpha: float
        '''

        param={
            'xai': xai,
            'epsilon': epsilon,
            'alpha': alpha
        }
        self.middle_size=encoder.middle_size

        xai_s=xai.split("_")[0] # short name for XAI [LRP, AGF, RAP]
        self.xai_s=xai_s
        mod_name=f"old_arch.{xai_s}_layers"
      

        # try to use getattr(importlib.import_module(args.cam_network), 'CAM')
        self.inv_linear_1=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.backbone.classifier[6], **param)
        self.inv_linear_2=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.backbone.classifier[3], **param)
        self.inv_linear_3=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.backbone.classifier[0], **param)
        # reshape the tensor
        self.inv_avgpool=getattr(implib(mod_name), f'{xai_s}_avgpool2d')(encoder.backbone.avgpool, **param)
        self.deconv = nn.Sequential(
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.features[30], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[28], **param),       
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[26], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[24], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.features[23], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[21], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[19], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[17], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.features[16], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[14], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[12], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[10], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.features[9], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[7], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[5], **param),
         
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.features[4], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[2], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.features[0], **param)
        )


    def forward(self, x, class_idx):
        '''
        x: torch.tensor, initialized tensor for decoder, shape: N x C
        class_idx, int
        '''
        if self.xai_s in ["LRP", "RAP"]:
            x = self.inv_linear_1(x)
            x = self.inv_linear_3(self.inv_linear_2(x))
            x = x.view(-1, *tuple(self.middle_size[1:])) # use view rather than reshape
            x = self.inv_avgpool(x)
            x = self.deconv(x)
            return torch.sum(x,dim=1)
        else:
            cam, grad_outputs = self.inv_linear_1(x, class_id=[class_idx,]*x.shape[0])
            cam, grad_outputs = self.inv_linear_3(self.inv_linear_2(cam, grad_outputs))

            cam=cam.view(-1, *tuple(self.middle_size[1:])) #only reshape cam element
            grad_outputs = grad_outputs.view(-1, *tuple(self.middle_size[1:]))

            cam, grad_outputs = self.inv_avgpool( cam, grad_outputs)
       
            x=(cam, grad_outputs)
            cam, grad_outputs = self.deconv(x)

            cam = cam / minmax_dims(cam, 'max') # shape N x 3 xH xW
            # return cam.sum(1, keepdim=True)
            return cam.sum(1, keepdim=False)

