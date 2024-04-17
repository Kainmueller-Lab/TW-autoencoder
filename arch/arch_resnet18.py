# This py file is suitable for resnet18 and resnet34 which use the BaiscBlock module to build
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as  np
import sys
from .architectures_utils import *
from .AGF_layers import minmax_dims
from importlib import import_module as implib



def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())
# --------------------------------------
# Encoder and Decoder
# --------------------------------------

class CNN_Encoder(nn.Module):
    def __init__(self, backbone, img_channel=3, input_size=(3, 224, 224),num_classes=6):
        super(CNN_Encoder, self).__init__()
        assert backbone=="resnet18" or backbone=="resnet34", "The selected CNN_encoder module should have backbone = renset18 renset34 "
        self.backbone = initialize_pretrainmodel(backbone, num_classes, img_channel, feature_extract=False, use_pretrained=True) 
        
        # self.flat_fts=int(np.prod(self.middle_size[1:]))
        

    def forward(self, x): 
        x=self.backbone(x) 
        return x 
        
    
class CNN_Decoder(nn.Module):
    def __init__(self, encoder, xai, epsilon, alpha, memory_efficient, **kwargs):
        super(CNN_Decoder, self).__init__()
        '''
        encoder: nn. Mondule,
        xai: str,
        epsilon: float
        alpha: float
        memory_efficient: bool

        reference:
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        '''
        self.normal_relu=kwargs.pop('normal_relu',False)
        self.normal_deconv=kwargs.pop('normal_deconv',False) 
        self.multiply_input=kwargs.pop('multiply_input',False)
        if self.normal_relu or self.normal_deconv or self.multiply_input:
            assert xai=="LRP_epsilon", "only support LRP_epsilon for ablation test"

        param={
            'xai': xai,
            'epsilon': epsilon,
            'alpha': alpha,
            'memory_efficient': memory_efficient,
            'detach_bias': kwargs.pop('detach_bias',False), 
            'normal_relu':kwargs.pop('normal_relu',False),
            'normal_deconv':kwargs.pop('normal_deconv',False),
            'normal_unpool':kwargs.pop('normal_unpool',False),
            'remove_heaviside':kwargs.pop('remove_heaviside',False),
            'multiply_input':kwargs.pop('multiply_input',False),
            'add_bottle_conv':kwargs.pop('add_bottle_conv',False)
            
        }
        self.param=param
        
        
        xai_s=xai.split("_")[0] # short name for XAI [LRP, AGF, RAP]
        # method cLRP also belongs to LRP module
        if xai_s=="cLRP":
            xai_s="LRP" 
        self.xai_s=xai_s
        mod_name=f"arch.{xai_s}_layers"
        self.mod_name=mod_name

       
       
    
        # try to use getattr(importlib.import_module(args.cam_network), 'CAM')
        self.inv_linear_1=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.backbone.fc, **param, top_layer=True)
       
        # reshape the tensor
        self.inv_avgpool=getattr(implib(mod_name), f'{xai_s}_avgpool2d')(encoder.backbone.avgpool, **param)
        self.avgpool_os=encoder.backbone.avgpool.output_size
        # layer blocj 4

        self.inv_layer4=self._make_inv_layer(encoder.backbone.layer4)
        
        self.inv_layer3=self._make_inv_layer(encoder.backbone.layer3)
        self.inv_layer2=self._make_inv_layer(encoder.backbone.layer2)
        self.inv_layer1=self._make_inv_layer(encoder.backbone.layer1)
        # first several layers

        
        self.inv_maxpool=  getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.backbone.maxpool, **param,)
        self.inv_bn1= getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.backbone.bn1, **param)
        self.inv_conv1= getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.backbone.conv1, **param)

    def forward(self, x, class_idx):
        '''
        x: torch.tensor, initialized tensor for decoder, shape: N x C
        class_idx, int
        '''
        if self.xai_s in ["LRP", "RAP"]:
            x = self.inv_linear_1(x)
        
            x = x.view(x.shape[0],-1,*self.avgpool_os) # use view rather than reshape
         
            x = self.inv_avgpool(x)

            x = self.inv_layer4(x)
            x = self.inv_layer3(x)
            x = self.inv_layer2(x)
            x = self.inv_layer1(x)

            x = self.inv_maxpool(x)

            # add for ablation test
            if self.normal_relu:
                x=F.relu(x)

            x = self.inv_bn1(x)
            x = self.inv_conv1(x)

            if not self.normal_deconv:
                if self.multiply_input:
                    x=torch.mul(x,self.inv_conv1.m.in_tensor)
                return torch.sum(x,dim=1)
            else:
                return x
        else:
            cam, grad_outputs = self.inv_linear_1(x, class_id=[class_idx,]*x.shape[0])
            cam, grad_outputs = self.inv_avgpool(cam, grad_outputs)

            cam, grad_outputs = self.inv_layer4( cam, grad_outputs)
            cam, grad_outputs = self.inv_layer3( cam, grad_outputs)
            cam, grad_outputs = self.inv_layer2( cam, grad_outputs)
            cam, grad_outputs = self.inv_layer1( cam, grad_outputs)
       
           
            cam, grad_outputs = self.inv_maxpool(x)
            cam, grad_outputs = self.inv_bn1(x)
            cam, grad_outputs = self.inv_conv1(x)

            cam = cam / minmax_dims(cam, 'max') # shape N x 3 xH xW
            # return cam.sum(1, keepdim=True)
            return cam.sum(1, keepdim=False)

        

    def _make_inv_layer(self, layer):
       
     
        blocks_list = [layer[i] for i in range(len(layer))]
        # print(f"The len of blocks list {len(blocks_list)}")
        inv_layer=[]
        
        for i, block in enumerate(blocks_list[::-1]):
            inv_layer.append(Inv_BasicBlock(self.mod_name, self.xai_s,block,self.param))
            # update the block_out_tensor which is the in_tensor of forward block
    
            
        

        return nn.Sequential(*inv_layer)

    def get_in_tensor(self, m):
        if hasattr(m, 'X'):
            in_tensor=m.X
        elif hasattr(m, 'in_tensor'):
            in_tensor=m.in_tensor
        else:
            raise RuntimeError(f'This layer {m} does not register in_tensor attribute')

        return in_tensor

class Inv_BasicBlock(nn.Module):
    def __init__(self,mod_name, xai_s, block , param):
        super().__init__()
        # print(block)
        self.normal_relu=param['normal_relu']
        if param['normal_relu']==True:
            # inplace=True means that it will modify the input directly, without allocating any additional output.
            self.relu=nn.ReLU(inplace=True)  
       

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
           
        
        if  self.normal_relu:
            self.inv_block=nn.Sequential(self.inv_bn2,self.inv_conv2,self.relu,
                                    self.inv_bn1,self.inv_conv1
                                    )
        else:
            self.inv_block=nn.Sequential( self.inv_bn2,self.inv_conv2,
                                    self.inv_bn1,self.inv_conv1
                                    )
        self.block=block

    def forward(self,x):
        if self.normal_relu:
            # here x is gradient
            x= self.relu(x)
            x1=self.inv_downsample(x)
            x2=self.inv_block(x)
        else:
        
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