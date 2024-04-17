import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as  np
import sys
#from .architectures_utils import *
import segmentation_models_pytorch as smp
import sys
# --------------------------------------
# Encoder and Decoder
# --------------------------------------

class CNN_Encoder(nn.Module):
    def __init__(self, img_channel=3,input_size=(3, 224, 224),num_classes=6):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size

        self.backbone = smp.encoders.get_encoder(
            name= "timm-efficientnet-b7",
            in_channels=3,
            depth=5,
            weights="imagenet")
        
        print(self.backbone)
        print(self.backbone.out_channels,"encoder outchannels")
        sys.exit()
        self.middle_size = self.get_flat_fts([self.backbone.features, self.backbone.avgpool])
        self.flat_fts=int(np.prod(self.middle_size[1:]))


    def get_flat_fts(self, fts):
        if isinstance(fts, (tuple, list)):
            f=Variable(torch.ones(1, *self.input_size))
            for ft in fts:               
                f = ft(f)
        else:
            f = fts(Variable(torch.ones(1, *self.input_size)))
        return f.size()
    
    def forward(self, x): 
        return self.backbone(x) 




#############################################################

class CNN_Decoder(nn.Module):
    def __init__(self, encoder):
        super(CNN_Decoder, self).__init__()
        self.middle_size=encoder.middle_size

        # inverse operation
        self.inv_linear_1=LRP_linear(encoder.backbone.classifier[6])
        self.inv_linear_2=LRP_linear(encoder.backbone.classifier[3])
        self.inv_linear_3=LRP_linear(encoder.backbone.classifier[0])
        # reshape the tensor
        self.inv_avgpool=LRP_avgpool2d(encoder.backbone.avgpool)
        self.deconv = nn.Sequential(
            LRP_maxpool2d(encoder.backbone.features[30]),
            LRP_transposeconv2d(encoder.backbone.features[28]),       
            LRP_transposeconv2d(encoder.backbone.features[26]),
            LRP_transposeconv2d(encoder.backbone.features[24]),
            
            LRP_maxpool2d(encoder.backbone.features[23]),
            LRP_transposeconv2d(encoder.backbone.features[21]),
            LRP_transposeconv2d(encoder.backbone.features[19]),
            LRP_transposeconv2d(encoder.backbone.features[17]),
            
            LRP_maxpool2d(encoder.backbone.features[16]),
            LRP_transposeconv2d(encoder.backbone.features[14]),
            LRP_transposeconv2d(encoder.backbone.features[12]),
            LRP_transposeconv2d(encoder.backbone.features[10]),
            
            LRP_maxpool2d(encoder.backbone.features[9]),
            LRP_transposeconv2d(encoder.backbone.features[7]),
            LRP_transposeconv2d(encoder.backbone.features[5]),
         
            LRP_maxpool2d(encoder.backbone.features[4]),
            LRP_transposeconv2d(encoder.backbone.features[2]),
            LRP_transposeconv2d(encoder.backbone.features[0])
        )

    def forward(self, x):
        x = self.inv_linear_1(x)
        x = self.inv_linear_3(self.inv_linear_2(x))
        x = x.view(-1, *tuple(self.middle_size[1:])) # use view rather than reshape
        x = self.inv_avgpool(x)
        x = self.deconv(x)
        return torch.sum(x,dim=1)
