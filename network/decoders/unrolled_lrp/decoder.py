import torch
from torch import nn
from typing import Any, Callable, List, Optional, Type, Union

from .vgg import Unrolled_VGG_Decoder
from .vggbn import Unrolled_VGGbn_Decoder
from .resnet import Unrolled_Resnet_Decoder

class Tied_weighted_decoder(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder:  Optional[nn.Module] = None,
            xai: str = "LRP_epsilon",
            epsilon: float = 1e-8,
            alpha: float = 1.0,
            detach_bias: bool = True,
            **kwargs
        ):
        super().__init__()
        if encoder_name.lower() == "vgg16":
            self.decoder = Unrolled_VGG_Decoder(
                encoder,
                xai,
                epsilon,
                alpha,
                detach_bias=detach_bias,
                **kwargs)
        elif encoder_name.lower() == "vgg16_bn":
            self.decoder = Unrolled_VGGbn_Decoder(
                encoder,
                xai,
                epsilon,
                alpha,
                detach_bias=detach_bias,
                **kwargs)
        elif "resnet" in encoder_name.lower():
            self.decoder = Unrolled_Resnet_Decoder(
                encoder,
                xai,
                epsilon,
                alpha,
                detach_bias=detach_bias,
                **kwargs)
        else:
            raise NotImplementedError(
                f"Unrolled LRP model not implemented for encoder {encoder_name}")

    def forward(self, x, class_idx):
        # class_idx is needed when xai is AGF 
        decoder_output = self.decoder(x, class_idx)
        return decoder_output
