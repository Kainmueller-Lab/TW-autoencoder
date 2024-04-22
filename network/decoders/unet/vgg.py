import torch
from torch import nn
from typing import cast

from ._utils import sequentialMultiInput

cfgs = {
    "A": [[64], [128, "M"],
          [256, 256, "M"],
          [512, 512, "M"],
          [512, 512, "M"],
          [512, 512, "M"]],
    "B": [[64, 64],
          [128, 128, "M"],
          [256, 256, "M"],
          [512, 512, "M"],
          [512, 512, "M"],
          [512,  512, "M"]],
    "D": [[64, 64],
          [128, 128, "M"],
          [256, 256, 256, "M"],
          [512, 512, 512, "M"],
          [512, 512, 512, "M"],
          [512, 512,"M"]],
    "E": [[64, 64],
          [128, 128, "M"],
          [256, 256, 256, 256, "M"],
          [512, 512, 512, 512, "M"],
          [512, 512, 512, 512, "M"],
          [512, 512,"M"]]
}


class VGG_Decoder(nn.Module):
    def __init__(self, encoder, use_batchnorm):
        super(VGG_Decoder, self).__init__()

        self.blocks = self._make_inv_layers(
            use_batchnorm, list(encoder.out_channels), cfgs["D"])

    def _make_inv_layers(self, batch_norm, skip_channels_list, cfg):
        blocks = []
        in_channels = int(skip_channels_list[-1])
        # center layers does not do need skip connections, so skip_channels_list[-1]=0
        skip_channels_list[-1] = 0

        for sub_cfg in cfg[::-1]:
            block = []
            for i, v in enumerate(sub_cfg):
                if v == "M":
                    block += [nn.Upsample(scale_factor=2, mode='nearest')]
                else:
                    v = cast(int, v)
                    if i == 0:
                        #should be a skip connection
                        conv2d = nn.Conv2d(
                            in_channels + skip_channels_list[-1],
                            v,
                            kernel_size=3,
                            padding=1)
                        del skip_channels_list[-1]
                    else:
                        conv2d = nn.Conv2d(
                            in_channels,
                            v,
                            kernel_size=3,
                            padding=1)
                    if batch_norm:
                        block += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        block += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v

            blocks.append(sequentialMultiInput(*block))

        # must use nn.ModuleList otherwise, it won't work and put the nn.module to cuda
        return nn.ModuleList(blocks)

    def forward(self, *features):

        bottleneck_feature = features[-1]

        # self.blocks[0] corresponds to central block
        x = self.blocks[0](bottleneck_feature)

        inv_features = features[:-1][::-1]
        for block, feature in zip(self.blocks[1:], inv_features):
            x = torch.cat([x, feature], dim=1)
            x = block(x)

        return x
