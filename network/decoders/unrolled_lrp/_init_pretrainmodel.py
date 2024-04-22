import torch
from torch import nn
from torchvision import  models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_pretrainmodel(model_name, num_classes, img_channel, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
  

    if model_name == "resnet50":
        """ Resnet50
        """
        
        model_ft = models.resnet50(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet101":
        """ Resnet101
        """
        
        model_ft = models.resnet101(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet18":
        """ Resnet18
        """
        
        model_ft = models.resnet18(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "resnet34":
        """ Resnet34
        """
        
        model_ft = models.resnet34(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        
    
    elif model_name == "vgg16_bn":
        """ VGG16_bn
        Be careful, expects (224,224) sized images 
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # model_ft.features[0] = nn.Conv2d(img_channel, 64, 3, 1, 1)

    elif model_name == "vgg16":
        """ VGG16
        Be careful, expects (224,224) sized images 
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        # model_ft.features[0] = nn.Conv2d(img_channel, 64, 3, 1, 1)
        

    elif model_name == "inceptionv3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained, **{'transform_input':False})
        #model_ft = inception_v3(pretrained=use_pretrained, **{'transform_input':False})
        #parameter'transform_input'--> If True, preprocesses the input according to the method 
        #with which it was trained on ImageNet. Default: False
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        #print(f"In_features: {num_ftrs}")
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        # Redefine input layer
        model_ft.Conv2d_1a_3x3 = models.inception.BasicConv2d(img_channel, 32, kernel_size=3, stride=2)



    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft