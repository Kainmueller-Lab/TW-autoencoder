import importlib
from importlib import import_module as implib
import torch
from torch import nn
from ._utils import minmax_dims, safe_divide

class Inv_BasicBlock(nn.Module):
    def __init__(self,mod_name, xai_s, block , param,  remove_last_relu=False):
        super().__init__()
        # print(block)
        self.param=param
       

        self.inv_bn2= getattr(implib(mod_name), f'{xai_s}_BN2d')(block.bn2, **param) 
        self.inv_conv2= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.conv2, **param)

        self.inv_bn1= getattr(implib(mod_name), f'{xai_s}_BN2d')(block.bn1, **param) 
        self.inv_conv1= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.conv1, **param)

        if block.downsample is not None:
            
            self.inv_downsample = nn.Sequential(
                getattr(implib(mod_name), f'{xai_s}_BN2d')(block.downsample[1], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.downsample[0], **param),
            )

        else:
            self.inv_downsample= nn.Identity()  
           
        
       
        self.inv_block=nn.Sequential(self.inv_bn2,self.inv_conv2,
                                self.inv_bn1,self.inv_conv1
                                )
        self.block=block

    def forward(self,x):
        if isinstance(self.inv_downsample, nn.Identity):
            sc_line_out=self.get_in_tensor(self.inv_conv1.m)
        
        else:
            sc_line_out=self.block.downsample(self.get_in_tensor(self.inv_conv1.m))

        total_out= self.inv_bn2.m(self.get_in_tensor(self.inv_bn2.m))+sc_line_out


        # skip connection line
        x1=safe_divide(sc_line_out,total_out)*x
        x1=self.inv_downsample(x1)

        # main line
        x2=safe_divide(total_out-sc_line_out, total_out)*x
        x2=self.inv_block(x2)
        # decompose the relevance according to the ratio between

        return x1+x2

    def get_in_tensor(self, m):
        if hasattr(m, 'X'):
            in_tensor=m.X
        elif hasattr(m, 'in_tensor'):
            in_tensor=m.in_tensor
            
        else:
            raise RuntimeError('This layer does not register in_tensor attribute')

        return in_tensor