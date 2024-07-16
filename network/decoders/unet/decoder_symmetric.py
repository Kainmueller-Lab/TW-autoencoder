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
            encoder
    ):
        super().__init__()
        if "vgg" in encoder_name.lower():
            self.decoder = VGG_Decoder(
                encoder,
                "bn" in encoder_name.lower())
        elif "resnet" in encoder_name.lower():
            self.decoder = Resnet_Decoder(
                encoder)
        else:
            raise NotImplementedError(
                f"sym. decoder not implemented for encoder {encoder_name}")

    def forward(self, *features):
      
        decoder_output = self.decoder(*features)

        return decoder_output
