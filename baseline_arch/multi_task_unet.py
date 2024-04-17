'''
multi task unet
reference:
https://github.com/PedroRASB/ISNet/blob/master/AlternativeModels/unetMultiTask.py
'''
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import sys
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from typing import Any, cast, Dict, List, Optional, Union
from ._utils import VGG_FS_Decoder,Resnet_FS_Decoder
from ._multi_task_unet2  import VGG_CBB_Decoder,Resnet_CBB_Decoder

# --------------------------------------
# Multi-task UNet
# --------------------------------------

class MTUNet(nn.Module):
    def __init__(self,img_channel, backbone,num_classes, add_classification, no_skip_connection, fully_symmetric_unet, concate_before_block) -> None:
        super().__init__()
      
        self.encoder= smp.encoders.get_encoder(
            name=backbone,
            in_channels=img_channel,
            depth=5,
            weights="imagenet",
        )
        assert fully_symmetric_unet *concate_before_block == False, "fully_symmetric_unet and concate_before_block can only be True at the same time"

        # fully_symmetric_unet is implemented in a way that tensor is concated before block
        if "vgg" in backbone:
            if fully_symmetric_unet:
                self.decoder=VGG_FS_Decoder(self.encoder,backbone, no_skip_connection)
            elif concate_before_block:
                self.decoder=VGG_CBB_Decoder(self.encoder,backbone, no_skip_connection)
            else:
                self.decoder=VGG_Decoder(self.encoder,backbone, no_skip_connection)
        elif "resnet" in backbone:
            if fully_symmetric_unet:
                self.decoder=Resnet_FS_Decoder(self.encoder,backbone, no_skip_connection)
            elif concate_before_block:
                self.decoder=Resnet_CBB_Decoder(self.encoder,backbone, no_skip_connection)
            else:
                self.decoder=Resnet_Decoder(self.encoder,backbone, no_skip_connection)
        else:
            raise NotImplementedError
        self.segmentation_head=SegmentationHead(in_channels=64, out_channels=num_classes)

        if add_classification==True:
            self.classification_head=ClassificationHead(backbone,num_classes)
        else:
            self.classification_head = None

        self.initialize()

        self.no_skip_connection=no_skip_connection
       

    def initialize(self):
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            self.initialize_head(self.classification_head)


    def initialize_decoder(self,module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def initialize_head(self,module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        # print(next(self.encoder.parameters()).is_cuda)
        # print(next(self.decoder.parameters()).is_cuda)
       
        features = self.encoder(x)
        if self.no_skip_connection:
            features[:-1]=[None]*(len(features)-1)

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])

          
            return masks, labels

        return masks


# --------------------------------------
# VGG_Decoder
# --------------------------------------


cfgs= {
    "A": [[64], [128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"],[512, 512, "M"]],
    "B": [[64, 64], [128, 128, "M"], [256, 256, "M"], [512, 512, "M"], [512, 512, "M"],[512,  512, "M"]],
    "D": [[64, 64], [128, 128, "M"], [256, 256, 256, "M"], [512, 512, 512, "M"], [512, 512, 512, "M"],[512, 512,"M"]],
    "E": [[64, 64], [128, 128, "M"], [256, 256, 256, 256, "M"], [512, 512, 512, 512, "M"], [512, 512, 512, 512, "M"], [512, 512,"M"]]}


class VGG_Decoder(nn.Module):
    def __init__(self, encoder, backbone, no_skip_connection):
        super(VGG_Decoder, self).__init__()

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
                        conv2d = nn.Conv2d(in_channels+skip_channels_list[-1], v, kernel_size=3, padding=1)
                        del skip_channels_list[-1]
                    else:
                        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        block += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        block += [conv2d, nn.ReLU(inplace=True)]
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

class Resnet_Decoder(nn.Module):
    def __init__(self, encoder, backbone, no_skip_connection):
        super(Resnet_Decoder, self).__init__()

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
        self.relu = nn.ReLU(inplace=True)


        

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
        # if bottle_up, only use the last block to do upsampling
        if bottle_up:
           
            if isinstance( layer[0], BasicBlock):
                inv_layer.append(Inv_BasicBlock(0,layer[0],skip_channels, outpadding))
            elif isinstance(layer[0] ,Bottleneck):
                inv_layer.append(Inv_Bottleneck(0,layer[0],skip_channels, outpadding))
            else:
                raise NotImplementedError
        
        else:
            skip_channels_list=[skip_channels]+[0,]*(len(layer)-1)
         
            blocks_list = [layer[i] for i in range(len(layer))]

            # inversely go through all layers and also do upsampling, skip_channels_list has no need to inverse
            for i, block, skip in zip(range(len(blocks_list)),blocks_list[::-1],skip_channels_list):
                if isinstance(block, BasicBlock):
                    inv_layer.append(Inv_BasicBlock(i, block,skip, outpadding))
                elif isinstance(block,Bottleneck):
                    inv_layer.append(Inv_Bottleneck(i, block,skip, outpadding))
                else:
                    raise NotImplementedError
        
                
        

        return sequentialMultiInput(*inv_layer)


# --------------------------------------
# Inv_BasicBlock  for resnet18,34
# --------------------------------------

class Inv_BasicBlock(nn.Module):
    def __init__(self,i, block, skip_channels,outpadding):
        super().__init__()
        '''
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        '''
        self.skip_channels=skip_channels

        # print(f"block index {i} and corresponding skip_channels {skip_channels}")
        

        self.conv1=nn.Conv2d(in_channels=block.conv2.out_channels+skip_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=3, stride=block.conv2.stride, padding=block.conv2.padding)
        self.bn1=nn.BatchNorm2d(block.conv2.in_channels)

        if block.downsample is not None:
            
            self.conv2=nn.ConvTranspose2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=block.conv1.kernel_size, stride=block.conv1.stride, padding=block.conv1.padding,
                                        output_padding=outpadding) #bias=None

            self.downsample=nn.Sequential(
                nn.ConvTranspose2d(in_channels=block.downsample[0].out_channels, out_channels=block.downsample[0].in_channels,
                                        kernel_size=block.downsample[0].kernel_size, stride=block.downsample[0].stride, padding=block.downsample[0].padding,
                                        output_padding=outpadding),
                nn.BatchNorm2d(block.downsample[0].in_channels)
            )

        else:
            
            self.conv2=nn.Conv2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=3, stride=block.conv1.stride, padding=block.conv1.padding)
            self.downsample=None 

        self.bn2=nn.BatchNorm2d(block.conv1.in_channels)

        self.relu = nn.ReLU(inplace=True)
      
        
            
    def forward(self,x,features=None):
        identity = x

        # use self.skip_channels as a switch to decide when to concate the features from UNet encoder
        if self.skip_channels>0 and features is not None:
            x = torch.cat([x, features], dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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

class Inv_Bottleneck(nn.Module):
    def __init__(self, i, block, skip_channels,outpadding):
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

        self.conv1 = nn.Conv2d(in_channels=block.conv3.out_channels+skip_channels, out_channels=block.conv3.in_channels,
                                kernel_size=1, stride=1)

        self.bn1=nn.BatchNorm2d(block.conv3.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # print(f"block index {i} and corresponding skip_channels {skip_channels}")

        if block.downsample is not None:
           
          
            self.conv2=nn.ConvTranspose2d(in_channels=block.conv2.out_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=block.conv2.kernel_size, stride=block.conv2.stride, padding=block.conv2.padding,output_padding=outpadding) #bias=None
            self.downsample=nn.Sequential(
                nn.ConvTranspose2d(in_channels=block.downsample[0].out_channels, out_channels=block.downsample[0].in_channels,
                                        kernel_size=block.downsample[0].kernel_size, stride=block.downsample[0].stride, padding=block.downsample[0].padding,
                                        output_padding=outpadding),
                nn.BatchNorm2d(block.downsample[0].in_channels)
            )
        else:
            
            self.conv2=nn.Conv2d(in_channels=block.conv2.out_channels, out_channels=block.conv2.in_channels,
                                        kernel_size=block.conv2.kernel_size, stride=block.conv2.stride, padding=block.conv2.padding)
            self.downsample=None 

        self.bn2=nn.BatchNorm2d(block.conv2.in_channels)
        self.conv3 = nn.Conv2d(in_channels=block.conv1.out_channels, out_channels=block.conv1.in_channels,
                                        kernel_size=1, stride=1)
        self.bn3=nn.BatchNorm2d(block.conv1.in_channels)



    def forward(self, x, features=None):
        identity = x
    
        # concatenation usually happens when the self.downsample = None
        # use self.skip_channels as a switch to decide when to concate the features from UNet encoder
        if self.skip_channels>0 and features is not None:
            x = torch.cat([x, features], dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out

       


class ClassificationHead(nn.Module):
    def __init__(self,backbone, num_classes):
        #Classification head attached to the bottleneck feature representation
        #ClassifierFeatures: number of elements before first dense layer
        #ClassificationClasses: number of classes
        super(ClassificationHead, self).__init__()
        self.backbone=backbone
        if "vgg" in backbone:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes),
            )
            
        elif "resnet" in backbone:
            if "18" in backbone or "34" in backbone:
                expansion=1
            elif "50" in backbone or "101" in backbone:
                expansion=4
            else:
                raise NotImplementedError
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * expansion, num_classes)
        else:
            raise NotImplementedError

    
    def vgg_cl_head_forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def resnet_cl_head_forwad(self,x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        if "vgg" in self.backbone:
            out=self.vgg_cl_head_forward(x)
        elif "resnet" in self.backbone:
            out=self.resnet_cl_head_forwad(x)
        else:
            raise NotImplementedError
        return out


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,  upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)