import torch.nn as nn
import torch
from .modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

        
class ClassificationHead(nn.Module):
    def __init__(self,encoder_name, num_classes):
        #Classification head attached to the bottleneck feature representation
        #ClassifierFeatures: number of elements before first dense layer
        #ClassificationClasses: number of classes
        super(ClassificationHead, self).__init__()
        self.encoder_name=encoder_name
        if "vgg" in encoder_name:
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
            
        elif "resnet" in encoder_name:
            if "18" in encoder_name or "34" in encoder_name:
                expansion=1
            elif "50" in encoder_name or "101" in encoder_name:
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
