import importlib
from importlib import import_module as implib
import torch
from torch import nn
import torch.nn.functional as F
from ._utils import minmax_dims, safe_divide

from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from ._inv_basicblock import Inv_BasicBlock
from ._inv_bottleneck import Inv_Bottleneck

class Unrolled_Resnet_Decoder(nn.Module):
    def __init__(self, encoder, xai, epsilon, alpha, **kwargs):
        super(Unrolled_Resnet_Decoder, self).__init__()
        '''
        encoder: nn. Mondule,
        xai: str,
        epsilon: float
        alpha: float
        memory_efficient: bool

        reference:
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        '''
        param={
            'xai': xai,
            'epsilon': epsilon,
            'alpha': alpha,
            'detach_bias': kwargs.pop('detach_bias',False)
            
        }
        self.param=param
        
        
        xai_s="LRP"
        mod_name=f"network.xai.{xai_s}_layers"
        self.mod_name=mod_name
        self.xai_s=xai_s

       
        self.bottleneck=encoder.avgpool
        self.bottle_conv = getattr(implib(mod_name),f'{xai_s}_bottle_conv')(encoder.fc,encoder.avgpool, **param)
        


        # layer 4 to layer 1
        self.inv_layer4=self._make_inv_layer(encoder.layer4,remove_last_relu=self.remove_last_relu)
        self.inv_layer3=self._make_inv_layer(encoder.layer3)
        self.inv_layer2=self._make_inv_layer(encoder.layer2)
        self.inv_layer1=self._make_inv_layer(encoder.layer1)
        # first several layers

        
        self.inv_maxpool=  getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.maxpool, **param,)
        self.inv_bn1= getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.bn1, **param)
        self.inv_conv1= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.conv1, **param)

    def forward(self, x, class_idx):
        '''
        x: torch.tensor, initialized tensor for decoder, shape: N x C
        class_idx, int
        '''
     
       
        x = self.bottleneck.in_tensor
        x = self.bottle_conv(x, class_idx)

        x = self.inv_layer4(x)
        x = self.inv_layer3(x)
        x = self.inv_layer2(x)
        x = self.inv_layer1(x)
        x = self.inv_maxpool(x)

        
        x = self.inv_bn1(x)
        x = self.inv_conv1(x)

        return torch.sum(x,dim=1)
      
      

        

    def _make_inv_layer(self, layer, remove_last_relu=False):
       
     
        blocks_list = [layer[i] for i in range(len(layer))]
        # print(f"The len of blocks list {len(blocks_list)}")
        inv_layer=[]
        
        for i, block in enumerate(blocks_list[::-1]):
            if isinstance(block, BasicBlock):

                if remove_last_relu== True and i==0:
                    inv_layer.append(Inv_BasicBlock(self.mod_name, self.xai_s,block,self.param, remove_last_relu=True))
                else:
                    inv_layer.append(Inv_BasicBlock(self.mod_name, self.xai_s,block,self.param))

            elif isinstance(layer[0] ,Bottleneck):

                if remove_last_relu== True and i==0:
                    inv_layer.append(Inv_Bottleneck(self.mod_name, self.xai_s,block,self.param, remove_last_relu=True))
                else:
                    inv_layer.append(Inv_Bottleneck(self.mod_name, self.xai_s,block,self.param))

            else:
                raise NotImplementedError

        return nn.Sequential(*inv_layer)
