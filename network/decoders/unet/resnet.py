import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from ._inv_basicblock import Inv_BasicBlock
from ._inv_bottleneck import Inv_Bottleneck
from ._utils import sequentialMultiInput
import torch.nn.functional as F

import sys
class Resnet_Decoder(nn.Module):
    def __init__(self, encoder, ablation_test, fcn):
        super(Resnet_Decoder, self).__init__()

        # add for ablation_test 
        skip_channels=encoder.out_channels
        if ablation_test and fcn: # reset skip_channels
            skip_channels=(*([0]*(len(encoder.out_channels)-1)),encoder.out_channels[-1])
            
        # ignore the layer4 as this is the bottleneck layer
        # self.inv_layer4=self._make_inv_layer(encoder.layer4)
        self.encoder = encoder

        self.inv_layer4 = self._make_inv_layer(
            encoder.layer4, 0, 1,  bottle_up=True)
        # self.inv_layer4 = self._make_inv_layer(
        #     encoder.layer4, 0, 1)
        self.inv_layer3 = self._make_inv_layer(
            encoder.layer3, skip_channels[-2], 1)
        self.inv_layer2 = self._make_inv_layer(
            encoder.layer2, skip_channels[-3], 1)
        # for the first layer, there is no downsampling
        self.inv_layer1 = self._make_inv_layer(
            encoder.layer1, skip_channels[-4], 0)

        self.inv_maxpool = nn.Upsample(scale_factor=2, mode='nearest')
        self.inv_conv1 = nn.ConvTranspose2d(
            in_channels=encoder.conv1.out_channels+skip_channels[-5],
            out_channels=64,
            kernel_size=encoder.conv1.kernel_size,
            stride=encoder.conv1.stride,
            padding=encoder.conv1.padding,
            output_padding=1)

    def forward(self, *features):

        bottleneck_feature = features[-1]

        x = self.inv_layer4(bottleneck_feature, None)
        x = self.inv_layer3(x, features[-2])
        x = self.inv_layer2(x, features[-3])
        x = self.inv_layer1(x, features[-4])
        x = self.inv_maxpool(x)

        if features[-5] is not None:
            x=torch.cat([x, features[-5]], dim=1)
        # x = F.relu(x) # this is not necessary

        x = self.inv_conv1(x)

        return x

    def _make_inv_layer(
            self, layer,num_skip_channels, outpadding, bottle_up=False):

        inv_layer = []
        # if bottle_up, only use the last block to do upsampling
        if bottle_up:
            if isinstance(layer[0], BasicBlock):
                inv_layer.append(
                    Inv_BasicBlock(
                        layer[0], num_skip_channels, outpadding))
            elif isinstance(layer[0] ,Bottleneck):
                inv_layer.append(
                    Inv_Bottleneck(
                        layer[0], num_skip_channels, outpadding))
            else:
                raise NotImplementedError

        else:
            skip_channels_list = [num_skip_channels] + [0,]*(len(layer)-1)

            blocks_list = [layer[i] for i in range(len(layer))]

            # inversely go through all layers and also do upsampling,
            # skip_channels_list has no need to inverse
            for i, (block, skip) in enumerate(
                    zip(blocks_list[::-1], skip_channels_list)):
                if isinstance(block, BasicBlock):
                    inv_layer.append(
                        Inv_BasicBlock(block, skip, outpadding))
                elif isinstance(block, Bottleneck):
                    inv_layer.append(
                        Inv_Bottleneck(block, skip, outpadding))
                else:
                    raise NotImplementedError

        return sequentialMultiInput(*inv_layer)
