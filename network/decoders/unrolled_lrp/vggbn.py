import importlib
from importlib import import_module as implib
import torch
from torch import nn
from ._utils import minmax_dims, safe_divide
import sys

class Unrolled_VGGbn_Decoder(nn.Module):
    def __init__(self, encoder, xai, epsilon, alpha, **kwargs):
        super(Unrolled_VGGbn_Decoder, self).__init__()
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
            'detach_bias': kwargs.pop('detach_bias',False), 
            'normal_relu':kwargs.pop('normal_relu',False),
            'normal_deconv':kwargs.pop('normal_deconv',False),
            'normal_unpool':kwargs.pop('normal_unpool',False),
            'remove_heaviside':kwargs.pop('remove_heaviside',False),
            'multiply_input':kwargs.pop('multiply_input',False),
            'add_bottle_conv':kwargs.pop('add_bottle_conv',False)
            
        }
        self.param=param
        self.remove_last_relu= kwargs.pop('remove_last_relu',False)
        self.ablation_test = kwargs.pop('ablation_test',False)

        if self.ablation_test:
            print(f"This is the ablation test for vgg16_bn backbone unrolled lrp model.")
        if self.param['normal_relu']:
            self.relu=nn.ReLU(True)
            assert xai=="LRP_epsilon", "only support LRP_epsilon for ablation test"

        self.avgpool_os=encoder.avgpool.output_size

        xai_s=xai.split("_")[0] # short name for XAI [LRP, AGF, RAP]
        # method cLRP also belongs to LRP module
        if xai_s=="cLRP":
            xai_s="LRP" 
        self.xai_s=xai_s
        mod_name=f"network.xai.{xai_s}_layers"
       

        # try to use getattr(importlib.import_module(args.cam_network), 'CAM')
        self.inv_linear_1=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[6], **param, top_layer=True)
        self.inv_linear_2=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[3], **param)
        self.inv_linear_3=getattr(implib(mod_name),f'{xai_s}_linear')(encoder.classifier[0], **param)
        # reshape the tensor
        self.inv_avgpool=getattr(implib(mod_name), f'{xai_s}_avgpool2d')(encoder.avgpool, **param)


        #TODO need to rewrite the following part
        if self.param['normal_relu']:
            if self.remove_last_relu:
                self.deconv = nn.Sequential(
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[43], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[41], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[40], **param), 
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[38], **param),      
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[37], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[35], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[34], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[33], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[31], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[30], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[28], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[27], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[25], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[24], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[23], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[21], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[20], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[18], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[17], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[15], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[14], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[13], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[11], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[10], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[8], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[7], **param),
            
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[6], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[4], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[3], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[1], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[0], **param)
            )
            else:
                self.deconv = nn.Sequential(
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[43], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[41], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[40], **param), 
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[38], **param),      
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[37], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[35], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[34], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[33], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[31], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[30], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[28], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[27], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[25], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[24], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[23], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[21], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[20], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[18], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[17], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[15], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[14], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[13], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[11], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[10], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[8], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[7], **param),
            
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[6], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[4], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[3], **param),
                self.relu,
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[1], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[0], **param)
            )
        else:
            self.deconv = nn.Sequential(
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[43], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[41], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[40], **param), 
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[38], **param),      
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[37], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[35], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[34], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[33], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[31], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[30], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[28], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[27], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[25], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[24], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[23], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[21], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[20], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[18], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[17], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[15], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[14], **param),
                
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[13], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[11], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[10], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[8], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[7], **param),
            
                getattr(implib(mod_name), f'{xai_s}_maxpool2d')(encoder.features[6], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[4], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[3], **param),
                getattr(implib(mod_name), f'{xai_s}_BN2d')(encoder.features[1], **param),
                getattr(implib(mod_name), f'{xai_s}_transposeconv2d')(encoder.features[0], **param)
            )


    def forward(self, x, class_idx):
        '''
        x: torch.tensor, initialized tensor for decoder, shape: N x C
        class_idx, int
        '''
      
        if self.param['normal_relu']:
            x = self.inv_linear_1(x)
            x = self.relu(x)
            x = self.inv_linear_2(x)
            x = self.relu(x)
            x = self.inv_linear_3(x) 
        else:
            x = self.inv_linear_1(x)
            x = self.inv_linear_3(self.inv_linear_2(x))
        
        x = x.view(x.shape[0],-1,*self.avgpool_os) # use view rather than reshape
        x = self.inv_avgpool(x)
        x = self.deconv(x)

        if self.param['multiply_input']:
            x=torch.mul(x,self.deconv[-1].m.in_tensor)
        
     