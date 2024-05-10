'''
symmetric decoder for resnet/vgg backbone
'''
import torch
from torch import nn

from .vgg import VGG_Decoder
from .resnet import Resnet_Decoder

class UnetDecoder_sym(nn.Module):
    def __init__(
            self,
            encoder_name,
            encoder,
            ablation_test,
            fcn
    ):
        super().__init__()
        self.ablation_test = ablation_test
        self.fcn = fcn
        if "vgg" in encoder_name.lower():
            self.decoder = VGG_Decoder(
                encoder,
                "bn" in encoder_name.lower())
        elif "resnet" in encoder_name.lower():
            self.decoder = Resnet_Decoder(
                encoder,
                ablation_test,
                fcn)
        else:
            raise NotImplementedError(
                f"sym. decoder not implemented for encoder {encoder_name}")

    def forward(self, *features):
        if self.ablation_test and self.fcn:
            new_features=list(features)
            new_features[:-1]=[None]*(len(features)-1) #except the bottleneck output, set all other features to be None
            decoder_output = self.decoder(*new_features)
        else:
            decoder_output = self.decoder(*features)

        return decoder_output
