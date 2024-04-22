import importlib
from importlib import import_module as implib
import torch
from torch import nn
from ._utils import minmax_dims, safe_divide

class Inv_Bottleneck(nn.Module):
    def __init__(self,mod_name, xai_s, block , param, remove_last_relu=False):
        super().__init__()
        # print(block)
        self.param=param
        self.remove_last_relu=remove_last_relu
        if self.param['normal_relu']==True:
            # inplace=True means that it will modify the input directly, without allocating any additional output.
            self.relu=nn.ReLU(inplace=True)  

        self.inv_bn3 = getattr(implib(mod_name), f'{xai_s}_BN2d')(block.bn3, **param) if not self.param['normal_deconv'] else nn.BatchNorm2d(block.conv3.in_channels)
        self.inv_conv3 = getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.conv3, **param)

        self.inv_bn2= getattr(implib(mod_name), f'{xai_s}_BN2d')(block.bn2, **param) if not self.param['normal_deconv'] else nn.BatchNorm2d(block.conv2.in_channels)
        self.inv_conv2= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.conv2, **param)

        self.inv_bn1= getattr(implib(mod_name), f'{xai_s}_BN2d')(block.bn1, **param) if not self.param['normal_deconv'] else nn.BatchNorm2d(block.conv1.in_channels)
        self.inv_conv1= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.conv1, **param)

        if block.downsample is not None:
            if not self.param['normal_deconv']: # when set normal deconv =False, use bn+conv
                self.inv_downsample = nn.Sequential(
                    getattr(implib(mod_name), f'{xai_s}_BN2d')(block.downsample[1], **param),
                    getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.downsample[0], **param),
                )
            else: # when set normal deconv =True, use conv+bn
                self.inv_downsample = nn.Sequential(
                    getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(block.downsample[0], **param),
                    nn.BatchNorm2d(block.downsample[0].in_channels),
                )

        else:
            self.inv_downsample= nn.Identity()  
           
        if  self.param['normal_relu'] and not self.param['normal_deconv']:
            self.inv_block=nn.Sequential(self.inv_bn3,self.inv_conv3,self.relu,
                                    self.inv_bn2,self.inv_conv2,self.relu,
                                    self.inv_bn1,self.inv_conv1
                                    )
        elif self.param['normal_relu'] and self.param['normal_deconv']: # when set normal deconv =True, use conv+bn
            self.inv_block=nn.Sequential(self.inv_conv3,self.inv_bn3,self.relu,
                                    self.inv_conv2,self.inv_bn2,self.relu,
                                    self.inv_conv1,self.inv_bn1
                                    )
        else:
            self.inv_block=nn.Sequential(self.inv_bn3,self.inv_conv3,
                                    self.inv_bn2,self.inv_conv2,
                                    self.inv_bn1,self.inv_conv1
                                    )
        self.block=block

    def forward(self,x):

        # if use the normal relu, the input relevance should first pass through relu
        if self.param['normal_relu']:
            if not self.remove_last_relu:
                x= self.relu(x)
            x1=self.inv_downsample(x)
            x2=self.inv_block(x)

        else:
        
            if isinstance(self.inv_downsample, nn.Identity):
                sc_line_out=self.get_in_tensor(self.inv_conv1.m)
            
            else:
                sc_line_out=self.block.downsample(self.get_in_tensor(self.inv_conv1.m))

            total_out= self.inv_bn3.m(self.get_in_tensor(self.inv_bn3.m))+sc_line_out


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