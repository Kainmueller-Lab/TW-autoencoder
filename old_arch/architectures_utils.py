import torch
from torch import nn
from torchvision import  models
# --------------------------------------
# Forward hook functions
# --------------------------------------



class FwdHooks:
    def __init__(self,xai):
        self.xai=xai
        self.allowed_pass_layers = (torch.nn.ReLU, torch.nn.ELU, 
                           torch.nn.Dropout, torch.nn.Dropout2d,
                           torch.nn.Dropout3d,
                           torch.nn.Softmax,
                           torch.nn.LogSoftmax,
                           torch.nn.Sigmoid,
                           torch.nn.Tanh,
                           torch.nn.LeakyReLU)  

    def get_layer_fwd_hook(self,layer):
        if self.xai=="AGF":
            return self.agf_fwd_hook

        elif self.xai=="RAP":
            return self.rap_fwd_hook
            
        elif "LRP" in self.xai:
            if isinstance(layer,
                        (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
                return self.max_pool_nd_fwd_hook

            if isinstance(layer,
                        (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            
                return self.conv_nd_fwd_hook

            if isinstance(layer, torch.nn.Linear):
                return self.linear_fwd_hook
            
            
            if isinstance(layer,(torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
                return self.batch_norm_nd_fwd_hook

            if isinstance(layer, (torch.nn.AdaptiveAvgPool2d,torch.nn.AdaptiveAvgPool1d)):
            
                return self.avgpool_fwd_hook

            if isinstance(layer, self.allowed_pass_layers):

                return self.silent_pass # must return a function in hook_func form!
            else:
                raise NotImplementedError("The network contains layers that"
                                        " are currently not supported {0:s}".format(str(layer)))
        else:
            raise NotImplementedError(f"The forward function for this {self.xai} method is not implemented yet")


    def rap_fwd_hook(self,m, input, output):
        if type(input[0]) in (list, tuple):
            m.X = []
            for i in input[0]:
                x = i.detach()
                x.requires_grad = True
                m.X.append(x)
        else:
            m.X = input[0].detach()
            m.X.requires_grad = True

        m.Y = output


    def agf_fwd_hook(self, m, input, output):
        if hasattr(m, 'X'):
            del m.X

        if hasattr(m, 'Y'):
            del m.Y

        m.reshape_gfn = None

        if type(input[0]) in (list, tuple):
            m.X = []
            for i in input[0]:
                x = i.detach()
                x.requires_grad = True
                m.X.append(x)
        else:
            m.X = input[0].detach()
            m.X.requires_grad = True

            if type(output) is torch.Tensor:
                if input[0].grad_fn is not None:
                    input_input = input[0].grad_fn(m.X)
                    if type(input_input) is torch.Tensor:
                        input_dims = input_input.dim()
                        output_dims = output.dim()
                        if input_dims != output_dims and input_dims == 4:
                            m.reshape_gfn = input[0].grad_fn

        m.Y = output

    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

        return
    def avgpool_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):

        # contain output size

        stride=(in_tensor[0].shape[-2]//m.output_size[-2],in_tensor[0].shape[-1]//m.output_size[-1])

        k0 = in_tensor[0].size()[-2] - (m.output_size[-2]-1)*stride[-2]
        k1 = in_tensor[0].size()[-1] - (m.output_size[-1]-1)*stride[-1]
        kernel_size=(k0,k1)
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'in_shape', in_tensor[0].size())
        setattr(m, 'stride', stride)
        setattr(m, 'kernel_size',kernel_size)
        setattr(m, 'in_channels',in_tensor[0].size()[1])


    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_tensor",  out_tensor)
        setattr(m, 'out_shape', list(out_tensor.size()))
        return
    
    
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_tensor",  out_tensor)
        setattr(m, "out_shape", list(out_tensor.size()))
        return
   
    def batch_norm_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):   
        
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_tensor', in_tensor[0])
        setattr(m, 'out_tensor', out_tensor)

    def silent_pass(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        #change by Xiaoyan
        pass
        # print("silent pass through ", m)
        
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())



# --------------------------------------
# Initialize encoder part layers
# --------------------------------------     


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
    
    elif model_name == "resnet50_PRM":
        """ Resnet50-PRM
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        backbone, num_features=get_backbone(model_ft)
        classifer=nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True)
        model_ft=nn.Sequential(backbone, classifier)

        
    
    elif model_name == "vgg16_bn":
        """ VGG16_bn
        Be careful, expects (224,224) sized images 
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        if use_pretrained==True:
            set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.features[0] = nn.Conv2d(img_channel, 64, 3, 1, 1)

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

    elif model_name == "inceptionv3_PRM":

        basebone=inception_v3_PRM(pretrained=False, **{'img_channel':img_channel, 'num_classes':num_classes,'transform_input':False})
        model_ft=peak_response_mapping(basebone,**PRM_config)
       

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
