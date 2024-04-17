import torch
from torch import nn
from torch.autograd import Variable
from .architectures_utils import *
import numpy as np
from .LRP_layers import *
# --------------------------------------
# Encoder and Decoder
# --------------------------------------

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28),num_classes=10):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16
        self.num_classes=num_classes
        self.embedding_size=output_size # 32


        # forward operations
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=1,
                     padding=1), # 1x16x27x27
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1), #1x32x13x13
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1), #1x64x6x6
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1), #1x128x3x3
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),#1x256x2x2
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.middle_size = self.get_flat_fts(self.conv)
        self.flat_fts=int(np.prod(self.middle_size[1:]))
        print("get the self.flat_fts")
        # reshape the tensor
        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size), # bacthnorm requires the bacth_size>1
            nn.LeakyReLU(0.2),
        )
        self.classifier=nn.Linear(self.embedding_size,num_classes)

    def get_flat_fts(self, fts):
        print(self.input_size,"xxxxxxyy")
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return f.size() 

    def forward(self, x): 
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)  
        x = self.linear(x)
        return self.classifier(x)


class CNN_Decoder(nn.Module):
    def __init__(self, encoder,input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.middle_size=encoder.middle_size

        # inverse operation
        self.inv_classifier=LRP_linear(encoder.classifier)
        self.inv_BN1d_1=LRP_BN1d(encoder.linear[1])
        self.inv_linear_1=LRP_linear(encoder.linear[0])
        # reshape the tensor
        self.deconv = nn.Sequential(
            LRP_BN2d(encoder.conv[12]),
            LRP_transposeconv2d(encoder.conv[11]),       
    
            LRP_BN2d(encoder.conv[9]),
            LRP_transposeconv2d(encoder.conv[8]),
         
            LRP_BN2d(encoder.conv[6]),
            LRP_transposeconv2d(encoder.conv[5]),
         
            LRP_BN2d(encoder.conv[3]),
            LRP_transposeconv2d(encoder.conv[2]),
            
            LRP_transposeconv2d(encoder.conv[0])
        )

    def forward(self, x):
        x = self.inv_classifier(x)
        x = self.inv_linear_1(self.inv_BN1d_1(x))
        x = x.view(-1, *tuple(self.middle_size[1:])) # use view rather than reshape
        x = self.deconv(x)
        return x

