import torch
from torch import nn

# --------------------------------------
# for resnet <= 34
# --------------------------------------
class Inv_BasicBlock(nn.Module):
    def __init__(self, block, skip_channels, outpadding):
        super().__init__()
        '''
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        '''
        self.skip_channels = skip_channels

        self.conv1 = nn.Conv2d(
            in_channels=block.conv2.out_channels+skip_channels,
            out_channels=block.conv2.in_channels,
            kernel_size=3,
            stride=block.conv2.stride,
            padding=block.conv2.padding)
        self.bn1 = nn.BatchNorm2d(block.conv2.in_channels)

        if block.downsample is not None:
            self.conv2 = nn.ConvTranspose2d(
                in_channels=block.conv1.out_channels,
                out_channels=block.conv1.in_channels,
                kernel_size=block.conv1.kernel_size,
                stride=block.conv1.stride,
                padding=block.conv1.padding,
                output_padding=outpadding) #bias=None

            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=block.downsample[0].out_channels,
                    out_channels=block.downsample[0].in_channels,
                    kernel_size=block.downsample[0].kernel_size,
                    stride=block.downsample[0].stride,
                    padding=block.downsample[0].padding,
                    output_padding=outpadding),
                nn.BatchNorm2d(block.downsample[0].in_channels)
            )

        else:
            self.conv2 = nn.Conv2d(
                in_channels=block.conv1.out_channels,
                out_channels=block.conv1.in_channels,
                kernel_size=3, stride=block.conv1.stride,
                padding=block.conv1.padding)
            self.downsample=None

        self.bn2 = nn.BatchNorm2d(block.conv1.in_channels)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x, features=None):
        identity = x

        # use self.skip_channels as a switch to decide when to concate
        # the features from UNet encoder
        if self.skip_channels > 0 and features is not None:
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
