# py file about constructing the equaivalent vgg-structure unet modle
from .unet import UNet
import torch
import torch.nn as nn


def initialize_weights(model):
    print("initialize the weights according to https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py")
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
#


def create_equivalent_unet_model(img_channel, backbone, input_size, num_classes):
    if backbone in ["vgg1","vggbn1"]:
        '''
        remove the last three FC layers and the last maxpooling layer, directly start the transpose upsampling and the decoder part
        '''
        unet_with_bn=True if backbone== "vggbn1" else False
        num_levels=5

        # Set the num_fmaps
        num_fmaps_down_list=[(64,64),(128,128),(256,256,256),(512,512,512),(512,512,512)]
        num_fmaps_up_list=[(512,512,256),(256,256,128),(128,64),(64,64)][::-1]



        kernel_size_unit=(3,3)
        level_conv_times=[2,2,3,3,3]
        assert len(level_conv_times)==num_levels

        # calculate the kernel_size_down
        kernel_size_down=[]
        kernel_size_up=[]
        for i in range(num_levels):
            kernel_size_down.append([kernel_size_unit]*level_conv_times[i])
   
        kernel_size_up=kernel_size_down[:-1]
      

   
        # calculate downsampling factors
        downsample_factor_unit=(2,2)
        downsample_factors=[downsample_factor_unit]*(num_levels-1)



        last_conv=torch.nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1, bias=True)

        model=torch.nn.Sequential(UNet(img_channel,
                                num_fmaps_down_list,
                                num_fmaps_up_list,
                                downsample_factors=downsample_factors,
                                kernel_size_down=kernel_size_down,
                                kernel_size_up=kernel_size_up,
                                activation='ReLU',
                                padding='same',
                                bn=unet_with_bn,
                                constant_upsample=False
                                ),
                                last_conv)
      
        # model initialization weights
        initialize_weights(model)

        print(f"The unet baseline model with backbone={backbone} is as follows:")
        # print(model)
        return model

    elif backbone in ["vgg2","vggbn2"]:
        '''
        remove the last three FC layers and then add two convs with first conv(kenerl_size 7x7(8x8),c=4096) and 
        second conv(kenerl_size 1x1,c=1000) and directly start the transpose upsampling(8x8, c-256) and the decoder part 
        '''
        unet_with_bn=True if backbone== "vggbn2" else False

        num_levels=6

        # Set the num_fmaps
        num_fmaps_down_list=[(64,64),(128,128),(256,256,256),(512,512,512),(512,512,512),(4096,1000,512)]
        num_fmaps_up_list=[(512,512,512),(512,512,256),(256,256,128),(128,64),(64,64)][::-1]


        # set the kernel size unit
        kernel_size_unit1=(3,3)
        
        if input_size[-2:]==(256,256):
            kernel_size_unit2=(8,8) 
        elif input_size[-2:]==(224,224):
            kernel_size_unit2=(7,7)
        else:
            raise NotImplementedError

        level_conv_times=[2,2,3,3,3,3]
        assert len(level_conv_times)==num_levels

        # calculate the kernel_size_down and up
        kernel_size_down=[]
        kernel_size_up=[]
        for i in range(num_levels-1):
            kernel_size_down.append([kernel_size_unit1]*level_conv_times[i])
        kernel_size_down.append([kernel_size_unit2,(1,1),kernel_size_unit2])


        kernel_size_up=kernel_size_down[:-1]


        # calculate downsampling factors
        downsample_factor_unit=(2,2)
        downsample_factors=[downsample_factor_unit]*(num_levels-1)



        last_conv=torch.nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1, bias=True)

        model=torch.nn.Sequential(UNet(img_channel,
                                num_fmaps_down_list,
                                num_fmaps_up_list,
                                downsample_factors=downsample_factors,
                                kernel_size_down=kernel_size_down,
                                kernel_size_up=kernel_size_up,
                                activation='ReLU',
                                padding='same',
                                bn=unet_with_bn,
                                constant_upsample=False
                                ),
                                last_conv)

        # model initialization weights
        initialize_weights(model)

        print(f"The unet baseline model with backbone={backbone} is as follows:")
        print(model)

        return model

    else:
        raise NotImplementedError
