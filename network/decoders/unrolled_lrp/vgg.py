import importlib
from importlib import import_module as implib
import torch
from torch import nn
from ._utils import minmax_dims, safe_divide

class Unrolled_VGG_Decoder(nn.Module):
    def __init__(self, encoder, xai, epsilon, alpha,**kwargs):
        super(Unrolled_VGG_Decoder, self).__init__()
        '''
        encoder: nn. Mondule,
        xai: str,
        epsilon: float
        alpha: float
        memory_efficient: bool
        '''
        param={
            'xai': xai,
            'epsilon': epsilon,
            'alpha': alpha,
            'detach_bias': kwargs.pop('detach_bias',False)    
        }
        self.param=param
      
        self.avgpool_os=encoder.avgpool.output_size

        xai_s="LRP"
        mod_name=f"network.xai.{xai_s}_layers"

        # try to use getattr(importlib.import_module(args.cam_network), 'CAM')
        self.inv_linear_1=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[6], **param, top_layer=True)
        self.inv_linear_2=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[3], **param)
        self.inv_linear_3=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[0], **param)
        # reshape the tensor
        self.inv_avgpool=getattr(implib(mod_name), f'{xai_s}_avgpool2d')(encoder.avgpool, **param)
        

        #TODO need to rewrite the following part
       
        self.deconv = nn.Sequential(
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[30], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[28], **param),       
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[26], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[24], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[23], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[21], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[19], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[17], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[16], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[14], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[12], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[10], **param),
            
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[9], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[7], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[5], **param),
        
            getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[4], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[2], **param),
            getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[0], **param)
        )


    def forward(self, x, class_idx):
        '''
        x: torch.tensor, initialized tensor for decoder, shape: N x C
        class_idx, int
        '''
       
        x = self.inv_linear_1(x)
        x = self.inv_linear_3(self.inv_linear_2(x))
        x = x.view(x.shape[0],-1,*self.avgpool_os) # use view rather than reshape
        x = self.inv_avgpool(x)
        x = self.deconv(x)

        return torch.sum(x,dim=1)
    