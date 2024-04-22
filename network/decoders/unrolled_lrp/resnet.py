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
            'detach_bias': kwargs.pop('detach_bias',False), 
            'normal_relu':kwargs.pop('normal_relu',False),
            'normal_deconv':kwargs.pop('normal_deconv',False),
            'normal_unpool':kwargs.pop('normal_unpool',False),
            'remove_heaviside':kwargs.pop('remove_heaviside',False),
            'multiply_input':kwargs.pop('multiply_input',False),
            'add_bottle_conv':kwargs.pop('add_bottle_conv',False)
            
        }
        self.param=param
        # self.remove_last_relu should not be put into the param dict
        self.ablation_test = kwargs.pop('ablation_test',False)
        self.remove_last_relu = kwargs.pop('remove_last_relu',False)

        if self.ablation_test:
            print(f"This is the ablation test for resnet backbone unrolled lrp model.")
        if self.param['normal_relu'] :
            assert xai=="LRP_epsilon", "only support LRP_epsilon for ablation."
        
        
        xai_s="LRP"
        mod_name=f"network.xai.{xai_s}_layers"
        self.mod_name=mod_name
        self.xai_s=xai_s

       
        self.bottleneck=encoder.avgpool
        self.bottle_conv = getattr(implib(mod_name),f'{xai_s}_bottle_conv')(encoder.fc,encoder.avgpool, **param)
        # if self.ablation_test:
        #     self.bottleneck=encoder.avgpool
        #     self.bottle_conv = getattr(implib(mod_name),f'{xai_s}_bottle_conv')(encoder.fc,encoder.avgpool, **param)
        # else:  
        #     # try to use getattr(importlib.import_module(args.cam_network), 'CAM')
        #     self.inv_linear_1=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.fc, **param, top_layer=True)
        #     self.inv_avgpool=getattr(implib(mod_name), f'{xai_s}_avgpool2d')(encoder.avgpool, **param)
        #     self.avgpool_os=encoder.avgpool.output_size
       


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
     
        if self.param['normal_relu'] and not self.param['normal_deconv']:
            x = self.bottleneck.in_tensor
            if not self.param["remove_heaviside"]: #default: with heaviside
                # x=torch.heaviside(x, values=torch.tensor([0.0]).to(self.device))
                x = safe_divide(x,x+self.param['epsilon'])
            x = self.bottle_conv(x, class_idx)
        
        elif self.param['normal_relu'] and self.param['normal_deconv']:
            # no heaviside function
            x = self.bottleneck.in_tensor
            x = self.bottle_conv(x,class_idx)

        else:
            ################# option 1 for unrolled lrp
            x = self.bottleneck.in_tensor
            x = self.bottle_conv(x, class_idx)

            ################# option 2 for unrolled lrp
            # x = self.inv_linear_1(x)
            # x = x.view(x.shape[0],-1,*self.avgpool_os) # use view rather than reshape
            # x = self.inv_avgpool(x)

        x = self.inv_layer4(x)
        x = self.inv_layer3(x)
        x = self.inv_layer2(x)
        x = self.inv_layer1(x)
        x = self.inv_maxpool(x)

        # add for ablation test
        # empirically found that if the model is more like unet, relu and inv_bn need to be remove
        if self.param['normal_relu']:
            x = F.relu(x)
        if not self.param['normal_deconv']:
            x = self.inv_bn1(x)

        x = self.inv_conv1(x)

        if not self.param['normal_deconv']:
            if self.param['multiply_input']:
                x=torch.mul(x,self.inv_conv1.m.in_tensor)
            return torch.sum(x,dim=1)
        else:
            return x
      

        

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
