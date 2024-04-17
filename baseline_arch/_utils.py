'''
fully symmetric unet decoder
'''
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import sys
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from typing import Any, cast, Dict, List, Optional, Union

cfgs= {
    "A": [[64], [128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"],[512, 512, "M"]],
    "B": [[64, 64], [128, 128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"],[512,  512, "M"]],
    "D": [[64, 64], [128, 128, "M"], [256, 256, 256, "M"], [512, 512, 512, "M"], [512, 512, 512, "M"],[512, 512,"M"]],
    "E": [[64, 64], [128, 128, "M"], [256, 256, 256, 256, "M"], [512, 512, 512, 512, "M"], [512, 512, 512, 512, "M"], [512, 512,"M"]]}


class VGG_FS_Decoder(nn.Module):
    def __init__(self, encoder, backbone, no_skip_connection):
        super(VGG_FS_Decoder, self).__init__()

        # print("----------------------")
        # print("-encoder  out channels-")
        # print(encoder.out_channels) #(64, 128, 256, 512, 512, 512)
        if no_skip_connection:
            skip_channels=(*([0]*(len(encoder.out_channels)-1)),encoder.out_channels[-1])
        else:
            skip_channels=encoder.out_channels  #(64, 128, 256, 512, 512, 512)

        batch_norm=True if "bn" in backbone else False
        self.blocks=self._make_inv_layers(batch_norm, list(skip_channels), cfgs["D"])



    def _make_inv_layers(self, batch_norm, skip_channels_list, cfg):
        blocks=[]
        in_channels=skip_channels_list[-1]
        del skip_channels_list[-1]
        # center layers does not do need skip connections, so skip_channels_list[-1]=0
        skip_channels_list.append(0)
        

        
        for sub_cfg in cfg[::-1]:
            block=[]
            for i, v in enumerate(sub_cfg):
                if v == "M":
                    block += [nn.Upsample(scale_factor=2,mode='nearest')]
                else:
                    v = cast(int, v)
                    if i==0:
                        #should be a skip connection
                        bacth_norm = nn.BatchNorm2d(v+skip_channels_list[-1])
                        conv2d = nn.Conv2d(in_channels+skip_channels_list[-1], v, kernel_size=3, padding=1)
                        del skip_channels_list[-1]
                    else:
                        bacth_norm = nn.BatchNorm2d(v)
                        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        block += [nn.ReLU(inplace=True), bacth_norm, conv2d]
                    else:
                        block += [nn.ReLU(inplace=True), conv2d]
                    in_channels = v

            blocks.append(sequentialMultiInput(*block))


        # must use nn.ModuleList otherwise, it won't work and put the nn.module to cuda
        return nn.ModuleList(blocks)

   
        

    def forward(self, *features):
     
        bottleneck_feature=features[-1]

        # self.blocks[0] corresponds to central block
        x = self.blocks[0](bottleneck_feature)


        inv_features=features[:-1][::-1]
        for block , feature in zip(self.blocks[1:],inv_features):
            x = torch.cat([x, feature], dim=1)
            x = block(x)

       



        return x

# --------------------------------------
# Resnet_Decoder
# --------------------------------------

class Resnet_FS_Decoder(nn.Module):
    def __init__(self, encoder, backbone, no_skip_connection):
        super(Resnet_FS_Decoder, self).__init__()

        # print("----------------------")
        # print("-encoder  out channels-")
        # print(encoder.out_channels)

        if no_skip_connection:
            skip_channels=(*([0]*(len(encoder.out_channels)-1)),encoder.out_channels[-1])
        else:
            skip_channels=encoder.out_channels 
        # ignore the layer4 as this is the bottleneck layer
        # self.inv_layer4=self._make_inv_layer(encoder.layer4)
        self.encoder=encoder
      
        self.inv_layer4=self._make_inv_layer(encoder.layer4, 0, outpadding=1, bottle_up=True)
   
    
        self.inv_layer3=self._make_inv_layer(encoder.layer3, skip_channels[-2],outpadding=1)
        self.inv_layer2=self._make_inv_layer(encoder.layer2, skip_channels[-3],outpadding=1)
        # for the first layer, there is no downsampling
        self.inv_layer1=self._make_inv_layer(encoder.layer1, skip_channels[-4],outpadding=0)
     
        
        self.inv_maxpool=nn.Upsample(scale_factor=2,mode='nearest')
        self.inv_conv1=nn.ConvTranspose2d(in_channels=encoder.conv1.out_channels+skip_channels[-5], out_channels=64,
                                        kernel_size=encoder.conv1.kernel_size, stride=encoder.conv1.stride, padding=encoder.conv1.padding,
                                        output_padding=1) #
     




    def forward(self, *features):

        bottleneck_feature=features[-1]

        x= self.inv_layer4(bottleneck_feature,None)
        x=self.inv_layer3(x,features[-2])
        x=self.inv_layer2(x,features[-3])
        x=self.inv_layer1(x,features[-4])
        x=self.inv_maxpool(x)

        if features[-5] is not None:
            x=torch.cat([x, features[-5]], dim=1)

        x=self.inv_conv1(x)

        return x

    def _make_inv_layer(self, layer,skip_channels, outpadding, bottle_up=False):
       
        # print("---------layer-----------")
        # print(layer)

        inv_layer=[]
        # if bottle_up, only use the first blok layer[0] to do upsampling
        if bottle_up:
           
            # if isinstance( layer[0], BasicBlock):
            #     inv_layer.append(Inv_FS_BasicBlock(0,layer[0],skip_channels, outpadding,remove_last_relu=True))
            # elif isinstance(layer[0] ,Bottleneck):
            #     inv_layer.append(Inv_FS_Bottleneck(0,layer[0],skip_channels, outpadding,remove_last_relu=True))
            # else:
            #     raise NotImplementedError
        

            # replace layer[0] by layer
            skip_channels_list=[skip_channels]+[0,]*(len(layer)-1)
            blocks_list = [layer[i] for i in range(len(layer))]

            #only remove the first relu for the bottle_up layer
            remove_last_relu_list=[True]+[False,]*(len(layer)-1)
            for i, block, skip, remove in zip(range(len(blocks_list)),blocks_list[::-1],skip_channels_list,remove_last_relu_list):
                if isinstance(block, BasicBlock):
                    inv_layer.append(Inv_FS_BasicBlock(i, block,skip, outpadding,remove_last_relu=remove))
                elif isinstance(block,Bottleneck):
                    inv_layer.append(Inv_FS_Bottleneck(i, block,skip, outpadding,remove_last_relu=remove))
                else:
                    raise NotImplementedError
        else:
            skip_channels_list=[skip_channels]+[0,]*(len(layer)-1)
         
            blocks_list = [layer[i] for i in range(len(layer))]

            # inversely go through all layers and also do upsampling, skip_channels_list has no need to inverse
            for i, block, skip in zip(range(len(blocks_list)),blocks_list[::-1],skip_channels_list):
                if isinstance(block, BasicBlock):
                    inv_layer.append(Inv_FS_BasicBlock(i, block,skip, outpadding))
                elif isinstance(block,Bottleneck):
                    inv_layer.append(Inv_FS_Bottleneck(i, block,skip, outpadding))
                else:
                    raise NotImplementedError
        
                
        

        return sequentialMultiInput(*inv_layer)


# --------------------------------------
# Inv_BasicBlock  for resnet18,34
# --------------------------------------

class Inv_FS_BasicBlock(nn.Module):
    def __init__(self,i, block, skip_channels,outpadding, remove_last_relu=False):
        super().__init__()
        '''
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        '''
        self.skip_channels=skip_channels
        self.remove_last_relu=remove_last_relu
      
        # print(f"block index {i} and corresponding skip_channels {skip_channels}")
        

        self.bn1=nn.BatchNorm2d(block.conv2.out_channels+skip_channels)
        self.conv1=nn.Conv2d(in_channels=block.conv2.out_channels+skip_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=3, stride=block.conv2.stride, padding=block.conv2.padding)
        

        # if block.downsample is not None, self.conv2 must use transposeconv2d (because there is the possibility to do upsampling here)
        if block.downsample is not None:
            
            self.conv2=nn.ConvTranspose2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=block.conv1.kernel_size, stride=block.conv1.stride, padding=block.conv1.padding,
                                        output_padding=outpadding) #bias=None

            self.downsample=nn.Sequential(
                nn.BatchNorm2d(block.downsample[0].out_channels+skip_channels),
                nn.ConvTranspose2d(in_channels=block.downsample[0].out_channels+skip_channels, out_channels=block.downsample[0].in_channels,
                                        kernel_size=block.downsample[0].kernel_size, stride=block.downsample[0].stride, padding=block.downsample[0].padding,
                                        output_padding=outpadding),
         
            )

        else:
            
            self.conv2=nn.Conv2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=3, stride=block.conv1.stride, padding=block.conv1.padding)
            self.downsample=None 

        self.bn2=nn.BatchNorm2d(block.conv1.in_channels)

        self.relu = nn.ReLU(inplace=True)
      
        
            
    def forward(self,x,features=None):
        if self.skip_channels>0 and features is not None:
            x = torch.cat([x, features], dim=1)
        
       

        if not self.remove_last_relu:
            x = self.relu(x)
        

        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)
        
        out = self.bn2(out)
        out = self.conv2(out)
        

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity

        return out

# --------------------------------------
# work around way for passing multiple inputs to nn.sequential(*layers)
# --------------------------------------

class sequentialMultiInput(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs


# --------------------------------------
# Inv_Bottleneck  for resnet50,101
# --------------------------------------

class Inv_FS_Bottleneck(nn.Module):
    def __init__(self, i, block, skip_channels,outpadding, remove_last_relu=False):
        super().__init__()
        
        '''
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        '''
        self.skip_channels=skip_channels
        self.remove_last_relu=remove_last_relu

        self.bn1=nn.BatchNorm2d(block.conv3.out_channels+skip_channels)
        self.conv1 = nn.ConvTranspose2d(in_channels=block.conv3.out_channels+skip_channels, out_channels=block.conv3.in_channels,
                           kernel_size=1, stride=1)

       
        self.relu = nn.ReLU(inplace=True)

        # print(f"block index {i} and corresponding skip_channels {skip_channels}")

        if block.downsample is not None:

            self.conv2=nn.ConvTranspose2d(in_channels=block.conv2.out_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=block.conv2.kernel_size, stride=block.conv2.stride, padding=block.conv2.padding,output_padding=outpadding) #bias=None
            self.downsample=nn.Sequential(
                nn.BatchNorm2d(block.downsample[0].out_channels+skip_channels),
                nn.ConvTranspose2d(in_channels=block.downsample[0].out_channels+skip_channels, out_channels=block.downsample[0].in_channels,
                                        kernel_size=block.downsample[0].kernel_size, stride=block.downsample[0].stride, padding=block.downsample[0].padding,
                                        output_padding=outpadding)
                
            )
        else:
            
            self.conv2=nn.ConvTranspose2d(in_channels=block.conv2.out_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=block.conv2.kernel_size, stride=block.conv2.stride, padding=block.conv2.padding)
            self.downsample=None 

        
        self.bn2=nn.BatchNorm2d(block.conv2.out_channels)
        self.bn3=nn.BatchNorm2d(block.conv1.out_channels)
        self.conv3 = nn.ConvTranspose2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=1, stride=1)
        



    def forward(self, x, features=None):
        if self.skip_channels>0 and features is not None:
            x = torch.cat([x, features], dim=1)

  

        if not self.remove_last_relu:
            x = self.relu(x)
  
        
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)                                                                                                                                                                     

        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
      

      
        out += identity

        return out




